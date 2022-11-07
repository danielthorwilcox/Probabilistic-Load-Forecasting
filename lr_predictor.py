#linear regression predictor for energy data
#ToDo: move this to FORECASTING: right now, using the detailed load distribution as features makes the regression perform too well
#in reality, this should be LR time-series forecasting: taking lagged 'total load actual' data, it should predict the next (n) values

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df["total load actual"], df.prediction),
            'rmse' : mean_squared_error(df["total load actual"], df.prediction) ** 0.5,
            'r2' : r2_score(df["total load actual"], df.prediction)}

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def build_baseline_model(df, test_ratio, target_col):
    X, y = feature_label_split(df, target_col="total load actual")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    print("coeffs: ", model.coef_)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result

load_data = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
data_nostamp = load_data.drop(columns="time",inplace=True)

df_baseline = build_baseline_model(load_data, 0.2, 'value')
baseline_metrics = calculate_metrics(df_baseline)
print(baseline_metrics)
#df_baseline.plot(title="Linear Regression predictions")
#plt.show()
