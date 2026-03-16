
⚠️ Disclaimer
Please Read Carefully:
Accuracy: The maximum prediction rate for Win/Loss/Draw outcomes is currently 0.6 (60%).
For Entertainment Only: This application is strictly for educational and entertainment purposes. It is a hobbyist ML project, not a financial tool.
No Gambling: Do not use this tool for betting, gambling, or any commercial purposes.
User Responsibility: The developers are not responsible for any financial losses or damages incurred. Any gambling activity conducted is at your own risk and responsibility.


Football Oracle ⚽️🔮

Football Oracle is a Machine Learning application that utilizes regression models to predict the outcome of football matches. By analyzing home and away team features under specific conditions, the app provides data-driven score expectations.

🚀 Features
ML-Powered Predictions: Uses stacked regression models to calculate expected goals.

Dual Runtime: Can be executed locally or via API calls.

Feature Engine: An integrated engine automatically extracts and processes historical team data based on team names.

🧠 How the Model Works
1. Data & Feature Engineering
The model retrieves features from historical performance data for both teams. These features are scaled and fed into the regression architecture. The engine file contains specific functions that map team names to the required feature sets.

2. Regression Strategy
While actual match scores are discrete values (ranging from 0-8), this model uses a stacked regression approach to output continuous expected values (floats).
Model Stacking: Multiple regression models are layered to refine the prediction.
Error Margin: The model operates with a Mean Absolute Error (MAE) of approximately 0.8.


3. Interpreting Results (Win/Loss/Draw)
Because the raw outputs are floating-point numbers, the following logic is applied to determine the final match result:
Rounding: Various rounding methods are tested to convert predicted scores into realistic goal counts.
Draw Threshold: To account for the MAE, we implement a threshold of 0.1 - 0.3.
If the difference between the two predicted scores is less than or equal to this threshold, the match is classified as a Draw.
If the difference is greater than the threshold, a Win or Loss is derived accordingly.

🛠 Usage
pip install -r requirements.txt

