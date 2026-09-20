# ============================================================
# forecasting.py
# Purpose: Project a simple future trend (expenses or savings)
#          based on a user's past financial records, stored in
#          the database over time via database.py.
#
# METHOD: Simple Linear Regression on (time -> value). This is
# intentionally basic - it fits a straight line through past
# data points and extends it forward by one step. This is NOT a
# sophisticated time-series model (like ARIMA/Prophet) - that
# would be overkill and hard to justify with so little data for
# a student project. We are upfront about this limitation.
#
# MINIMUM DATA REQUIREMENT: at least 3 historical records are
# required before forecasting - a trend through 1-2 points is
# meaningless and could just be noise, not a real trend.
# ============================================================

import numpy as np
from sklearn.linear_model import LinearRegression

MIN_RECORDS_REQUIRED = 3


def forecast_next_value(historical_values: list) -> dict:
    """
    Given a list of past values (e.g. monthly expenses over the
    last few records, oldest first), fits a straight line and
    projects the next value forward.

    Returns a dict with the forecast, the trend direction, and a
    flag indicating whether forecasting was actually possible.
    """
    n = len(historical_values)

    if n < MIN_RECORDS_REQUIRED:
        return {
            "forecast_possible": False,
            "message": (
                f"Not enough historical data to forecast reliably. "
                f"You have {n} record(s); at least {MIN_RECORDS_REQUIRED} "
                f"are needed to identify a trend."
            ),
            "forecast_value": None,
            "trend_direction": None,
        }

    # X = time steps (0, 1, 2, ... n-1), representing each past record in order
    # y = the actual values at each of those time steps
    X = np.arange(n).reshape(-1, 1)
    y = np.array(historical_values)

    model = LinearRegression()
    model.fit(X, y)

    # Predict the NEXT time step (n), i.e. one step beyond the data we have
    next_step = np.array([[n]])
    forecast_value = model.predict(next_step)[0]

    # The slope of the fitted line tells us the trend direction
    slope = model.coef_[0]
    if slope > (0.02 * np.mean(y)):       # rising by more than ~2% of the average value
        trend_direction = "Increasing"
    elif slope < -(0.02 * np.mean(y)):
        trend_direction = "Decreasing"
    else:
        trend_direction = "Stable"

    return {
        "forecast_possible": True,
        "message": f"Forecast based on {n} historical records.",
        "forecast_value": round(float(forecast_value), 2),
        "trend_direction": trend_direction,
        "slope_per_period": round(float(slope), 2),
    }


def forecast_financial_history(records: list) -> dict:
    """
    Takes a list of financial_records dicts (as returned by
    database.get_records_for_user, oldest first) and forecasts
    the next value for expenses, savings, and health score trend.
    """
    if not records:
        return {
            "expenses_forecast": {"forecast_possible": False, "message": "No historical records found."},
            "savings_forecast": {"forecast_possible": False, "message": "No historical records found."},
        }

    expense_history = [r["monthly_expenses"] for r in records if r.get("monthly_expenses") is not None]
    savings_history = [r["monthly_savings"] for r in records if r.get("monthly_savings") is not None]

    return {
        "expenses_forecast": forecast_next_value(expense_history),
        "savings_forecast": forecast_next_value(savings_history),
    }


# ------------------------------------------------------------
# Quick manual test - only runs when this file is executed
# directly (python src/forecasting.py)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Simulate a user's expense history over 5 months, gradually rising
    sample_expense_history = [15000, 15500, 16200, 17000, 17800]
    result = forecast_next_value(sample_expense_history)
    print("Expense forecast (5 months of data):")
    print(f"  {result}")

    # Simulate a user with too little history
    too_little_history = [15000, 15500]
    result2 = forecast_next_value(too_little_history)
    print("\nExpense forecast (only 2 months of data):")
    print(f"  {result2}")

    # Simulate a stable savings trend
    stable_savings = [5000, 5100, 4950, 5050, 5000]
    result3 = forecast_next_value(stable_savings)
    print("\nSavings forecast (stable trend):")
    print(f"  {result3}")