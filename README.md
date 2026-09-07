# NBA Shot Quality  & Spatial Scouting 
An interactive scouting dashboard built with **Streamlit**, **Plotly**, and **Scikit-Learn** that evaluates shot quality across the 2025-26 NBA season. 

Instead of judging players purely on whether a shot went in, this project uses a trained **HistGradientBoosting** model to calculate what *should* have happened: the Expected Field Goal percentage ($xFG\%$) and Expected Points ($xPTS$) based on shot mechanics, clock context and spatial geometry.
---

## Why I Built This
Box scores shooting percentages are often misleading. The NBA is heavily reliant on isolation basketball. A talented scorer is one that not only shoots with high efficiency but is able to replicate the exact same looks even under a heavy contest at the end of the shot clock. Advanced statistics can also highlight how certain players benefit from easy shot diets due to their team which can make their box score numbers inflated. 

I wanted to build a tool that lets me explore practical machine learning whilst working on something I care about. I have always loved basketball, and sports tech is a field I would like to pursue.
---

# How It Works
```text
[NBA Stats API]
|[Shotchart Detail (2025-26 NBA Season)]
↓
[Feature Engineering]
|Shot Angle: arctan2(LOC_X, LOC_Y)
|Distance: Euclidean scaling from tenths of a foot
|Action clustering:70+ raw action types -> 8 archetypes
↓
[HistGradientBoostingClassifier]
|Trained on spatial & clock features
|Evaluated with Log Loss & Brier Score (probability calibration)
|Outputs expected conversion (xFG%) per attempt
↓
[Interactive dashboard]
|Plotly half-court with click-to-inspect shot markers 
|2K-style difficulty & contest meters
|Zone efficiency audit (Actual vs Expected)
```
---

## Key Metrics Explained
| Metric | Formula | What It Tells Us |
| :--- | :--- | :--- |
| **xFG%** | $P(\text{Make} \mid \text{Distance, Angle, Action, Clock})$ | The baseline likelihood of an average NBA player making this exact shot. |
| **Shooting Delta** | $\text{Actual FG\%} - \text{Expected xFG\%}$ | Positive values indicate above-average shot creation/making; negative values suggest underperformance. |
| **Expected Points (xPTS)** | $xFG\% \times \text{Shot Value (2 or 3)}$ | Evaluates possession yield. Demonstrates why a 36% corner three ($1.08\text{ xPTS}$) beats a 45% long midrange two ($0.90\text{ xPTS}$). |

---

## Technical Challenges & Engineering Trade-offs

### 1. Classification Accuracy vs. Probability Calibration
Early on, evaluating the model with standard classification accuracy was unhelpful—most basketball shots are missed, so a naive model predicting zero every time can hit ~53% accuracy while being completely useless. I shifted the optimization to **Log Loss** and the **Brier Score**, ensuring the model outputs reliable probabilities ($xFG\%$) rather than hard binary predictions.

### 2. Feature Engineering Around the Rim
NBA API spatial coordinates (`LOC_X`, `LOC_Y`) are recorded in tenths of a foot, with $(0, 0)$ placed at the center of the hoop. I calculated:
* Euclidean distance: $\frac{\sqrt{x^2 + y^2}}{10}$
* Directional shot angle: $\arctan2(x, y)$

Angle is essential because corner threes and straight-on threes have different baseline conversion rates despite identical distances. Additionally, raw tracking records over 70 distinct shot descriptions (e.g., *"Turnaround Fadeaway Bank Jump Shot"*); I clustered these into 8 distinct categorical archetypes (Dunks, Layups, Floaters, Pullups, Fadeaways, etc.) to give the gradient booster clear categorical boundaries without sparse overfitting.

### 3. Dropping the Video Stream for Real-Time Analytics
Initially, I attempted to stream broadcast replays inside the app using the NBA's internal `videoeventsasset` CDN endpoint. However, modern Akamai bot mitigation on `stats.nba.com` aggressively throttles or drops non-browser TLS handshakes, causing 6–10 second connection timeouts. Rather than leaving an unreliable feature that freezes the UI, I pivoted the right column into an **interactive Shot Quality HUD and Shot Diet audit**, using Streamlit's `on_select` state to inspect shot difficulty instantly when clicking court markers.

---

## Project Structure

```text
├── app.py                  # Main Streamlit dashboard & Plotly court layout
├── fetch_clean_data.py     # NBA API ingestion & feature engineering pipeline
├── train_model.py          # Model training, calibration, and artifact export
├── shots_cleaned.csv       # Processed dataset with engineered features
├── shot_success_model.pkl  # Serialized HistGradientBoosting model
├── requirements.txt        # Pinned project dependencies
└── .gitignore              # Keeps virtualenvs and scratchpads off GitHub
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/](https://github.com/)<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```