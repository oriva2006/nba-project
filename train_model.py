import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def load_data(file_path="shots_cleaned.csv"):
    """
    Loads the cleaned shot data from a CSV file.
    """
    return pd.read_csv(file_path)


def prepare_features(df):
    """
    Prepares features, target and game groups for modeling.
    """
    numerical_cols = [
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "SHOT_DISTANCE",
        "CALCULATED_DIST",
        "SHOT_ANGLE",
        "SHOT_VALUE"
    ]
    categorical_cols = ["SHOT_ZONE_BASIC", "SHOT_TYPE", "ACTION_GROUP"]

    X = df[numerical_cols + categorical_cols]
    y = df["TARGET"]
    groups = df["GAME_ID"]  # used to keep shots from one game on the same side of the split

    return X, y, groups, numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    # Scaling has no effect on tree models; it is here for the logistic regression baseline.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def evaluate(y_true, p):
    return {
        "log_loss": float(log_loss(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
    }


def reliability_table(y_true, p, n_bins=10):
    """
    Bins predictions into equal-sized groups and compares the average predicted
    probability with the actual make rate in each group.
    """
    bins = pd.qcut(p, q=n_bins, duplicates="drop")
    table = (
        pd.DataFrame({"p": p, "y": np.asarray(y_true), "bin": bins})
        .groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
        .reset_index(drop=True)
    )
    ece = float((table["n"] * (table["predicted"] - table["actual"]).abs()).sum() / table["n"].sum())
    return table, ece


def save_reliability_plot(table, path="reliability_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping reliability plot: pip install matplotlib)")
        return

    top = max(table["predicted"].max(), table["actual"].max()) * 1.05
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, top], [0, top], "--", color="grey", label="Perfect calibration")
    ax.plot(table["predicted"], table["actual"], "o-", label="Model")
    ax.set_xlabel("Predicted make probability")
    ax.set_ylabel("Actual make rate")
    ax.set_title("Reliability curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved reliability plot to {path}")


def train_model(X, y, groups, numerical_cols, categorical_cols):
    """
    Trains the shot-success model, compares it with baselines and reports evaluation metrics.
    """
    # Split by game so shots from the same game never appear in both train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {len(X_train)} shots in {groups.iloc[train_idx].nunique()} games | "
          f"Test: {len(X_test)} shots in {groups.iloc[test_idx].nunique()} games")
    print(f"Make rate: train {y_train.mean():.3f} | test {y_test.mean():.3f}")
    always_miss_acc = float(1 - y_test.mean())
    print(f"Accuracy of always predicting 'miss' on test: {always_miss_acc:.3f}")

    # Baseline 1: predict the training make rate for every shot
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)

    # Baseline 2: logistic regression on the same features
    logreg = Pipeline(steps=[
        ("preprocessor", build_preprocessor(numerical_cols, categorical_cols)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]).fit(X_train, y_train)

    # Main model
    model_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(numerical_cols, categorical_cols)),
        ("classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]).fit(X_train, y_train)

    p_base = baseline.predict_proba(X_test)[:, 1]
    p_lr = logreg.predict_proba(X_test)[:, 1]
    p_hgb = model_pipeline.predict_proba(X_test)[:, 1]

    results = {
        "baseline": evaluate(y_test, p_base),
        "logistic_regression": evaluate(y_test, p_lr),
        "hist_gradient_boosting": evaluate(y_test, p_hgb),
    }

    print("\n✅ Test-set comparison (lower log loss / Brier is better, higher AUC is better)")
    print(f"{'Model':<26}{'Log loss':>10}{'Brier':>10}{'ROC AUC':>10}")
    labels = {
        "baseline": "Baseline (make rate)",
        "logistic_regression": "Logistic regression",
        "hist_gradient_boosting": "HistGradientBoosting",
    }
    for key, label in labels.items():
        m = results[key]
        print(f"{label:<26}{m['log_loss']:>10.4f}{m['brier']:>10.4f}{m['roc_auc']:>10.4f}")

    base, lr, hgb = results["baseline"], results["logistic_regression"], results["hist_gradient_boosting"]
    ll_gain = 1 - hgb["log_loss"] / base["log_loss"]
    brier_skill = 1 - hgb["brier"] / base["brier"]
    ll_gain_vs_lr = 1 - hgb["log_loss"] / lr["log_loss"]
    print(f"\nLog loss reduction vs baseline: {ll_gain:.1%} | vs logistic regression: {ll_gain_vs_lr:.1%}")
    print(f"Brier skill score vs baseline:  {brier_skill:.1%}")

    # Overfitting check: train vs test log loss
    train_ll = float(log_loss(y_train, model_pipeline.predict_proba(X_train)[:, 1]))
    print(f"\nHGB log loss: train {train_ll:.4f} vs test {hgb['log_loss']:.4f} "
          f"(gap {hgb['log_loss'] - train_ll:+.4f})")

    # Stability check: 5-fold cross-validation, grouped by game
    cv_scores = -cross_val_score(
        clone(model_pipeline), X, y, groups=groups,
        cv=GroupKFold(n_splits=5), scoring="neg_log_loss",
    )
    print(f"5-fold grouped CV log loss: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Calibration check on the test set
    table, ece = reliability_table(y_test, p_hgb)
    print("\nReliability (test set, 10 equal-sized bins):")
    print(f"{'Predicted':>10}{'Actual':>10}{'Shots':>8}")
    for _, row in table.iterrows():
        print(f"{row['predicted']:>10.3f}{row['actual']:>10.3f}{int(row['n']):>8}")
    print(f"Expected calibration error (average gap, weighted by shots): {ece:.4f}")
    save_reliability_plot(table)

    metrics = {
        "n_shots": int(len(X)),
        "n_games": int(groups.nunique()),
        "test_make_rate": float(y_test.mean()),
        "always_miss_accuracy": always_miss_acc,
        "test": results,
        "log_loss_reduction_vs_baseline": float(ll_gain),
        "log_loss_reduction_vs_logreg": float(ll_gain_vs_lr),
        "brier_skill_score": float(brier_skill),
        "hgb_train_log_loss": train_ll,
        "cv_log_loss_mean": float(cv_scores.mean()),
        "cv_log_loss_std": float(cv_scores.std()),
        "expected_calibration_error": ece,
        "reliability_table": table.to_dict(orient="records"),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nSaved all metrics to metrics.json")

    return model_pipeline


def save_model(model, file_path="shot_success_model.pkl"):
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, file_path)
    print(f"✅ Model saved to {file_path}")


if __name__ == "__main__":
    df = load_data()
    X, y, groups, numerical_cols, categorical_cols = prepare_features(df)
    model = train_model(X, y, groups, numerical_cols, categorical_cols)
    save_model(model)