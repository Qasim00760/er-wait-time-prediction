"""Wait time prediction model.

Only information that is known when a patient arrives is used as input. The three
stage times (registration, triage, time to clinician) add up exactly to the total
wait, and patient outcome and satisfaction are recorded after the visit, so using
any of them would leak the answer into the model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.analysis import NAVY, BLUE, RED, STAGES, TARGET

CATEGORICAL = ["Urgency Level", "Time of Day", "Day of Week", "Season", "Region", "Hospital Name"]
NUMERIC = ["Nurse-to-Patient Ratio", "Specialist Availability", "Facility Size (Beds)", "Hour"]
FEATURES = CATEGORICAL + NUMERIC
LEAKY = STAGES + ["Patient Outcome", "Patient Satisfaction"]
RANDOM_STATE = 42


def _preprocessor(categorical=CATEGORICAL):
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)], remainder="passthrough"
    )


def candidate_models() -> dict:
    return {
        "Baseline (always predict the mean)": DummyRegressor(),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                           random_state=RANDOM_STATE),
    }


def make_pipeline(model, categorical=CATEGORICAL) -> Pipeline:
    return Pipeline([("prep", _preprocessor(categorical)), ("model", model)])


def split(df: pd.DataFrame, features=FEATURES):
    return train_test_split(df[features], df[TARGET], test_size=0.2, random_state=RANDOM_STATE)


def _scores(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Test-set scores and 5-fold cross-validated R2 for each candidate model."""
    X_train, X_test, y_train, y_test = split(df)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in candidate_models().items():
        pipe = make_pipeline(model).fit(X_train, y_train)
        scores = _scores(y_test, pipe.predict(X_test))
        cv_r2 = cross_val_score(make_pipeline(model), df[FEATURES], df[TARGET], cv=cv, scoring="r2")
        rows.append({"model": name, **scores, "CV_R2_mean": cv_r2.mean(), "CV_R2_std": cv_r2.std()})
    return pd.DataFrame(rows).round(3)


def leakage_demo(df: pd.DataFrame) -> pd.DataFrame:
    """Show what happens when the stage times and post-visit columns are used as features."""
    leaky_features = FEATURES + LEAKY
    categorical = CATEGORICAL + ["Patient Outcome"]
    X_train, X_test, y_train, y_test = split(df, leaky_features)
    rows = []
    for name, feats, cats in [("Arrival-time features only", FEATURES, CATEGORICAL),
                              ("With stage times, outcome and satisfaction", leaky_features, categorical)]:
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        pipe = make_pipeline(model, cats).fit(X_train[feats], y_train)
        rows.append({"features": name, **_scores(y_test, pipe.predict(X_test[feats]))})
    return pd.DataFrame(rows).round(3)


def train_final_model(df: pd.DataFrame) -> Pipeline:
    """Gradient Boosting trained on all rows, used by the app."""
    return make_pipeline(candidate_models()["Gradient Boosting"]).fit(df[FEATURES], df[TARGET])


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Permutation importance on the test set: how much MAE rises when a feature is shuffled."""
    X_train, X_test, y_train, y_test = split(df)
    pipe = make_pipeline(candidate_models()["Gradient Boosting"]).fit(X_train, y_train)
    result = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
                                    scoring="neg_mean_absolute_error")
    return (
        pd.DataFrame({"feature": FEATURES, "MAE_increase_min": result.importances_mean})
        .sort_values("MAE_increase_min", ascending=False)
        .round(2)
        .reset_index(drop=True)
    )


def test_predictions(df: pd.DataFrame) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = split(df)
    pipe = make_pipeline(candidate_models()["Gradient Boosting"]).fit(X_train, y_train)
    out = X_test.copy()
    out["actual"] = y_test
    out["predicted"] = pipe.predict(X_test).round(0)
    out["error"] = out["predicted"] - out["actual"]
    return out


# ---------------------------------------------------------------- charts


def plot_model_comparison(comparison: pd.DataFrame):
    c = comparison.set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(c.index, c["MAE"], color=[RED if "Baseline" in m else BLUE for m in c.index])
    axes[0].set_xlabel("Mean absolute error (minutes, lower is better)")
    axes[1].barh(c.index, c["R2"].clip(lower=0), color=[RED if "Baseline" in m else BLUE for m in c.index])
    axes[1].set_xlabel("R2 on test set (higher is better)")
    axes[1].set_xlim(0, 1)
    axes[1].set_yticklabels([])
    for ax, col, fmt in [(axes[0], "MAE", "{:.1f}"), (axes[1], "R2", "{:.3f}")]:
        for i, v in enumerate(c[col]):
            ax.text(max(v, 0) * 1.01 + 0.005, i, fmt.format(v), va="center", fontsize=10)
        ax.invert_yaxis()
        sns.despine(ax=ax)
    fig.suptitle("Model comparison on the same 20% test set", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(preds: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for level, part in preds.groupby("Urgency Level", observed=True):
        ax.scatter(part["actual"], part["predicted"], s=14, alpha=0.6, label=str(level))
    lim = [0, max(preds["actual"].max(), preds["predicted"].max()) * 1.05]
    ax.plot(lim, lim, color="#555", linestyle="--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual wait (minutes)")
    ax.set_ylabel("Predicted wait (minutes)")
    ax.legend(title="Urgency", frameon=False)
    return _finish(ax, "Gradient Boosting: actual vs predicted (test set)")


def plot_feature_importance(importance: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(importance["feature"], importance["MAE_increase_min"], color=NAVY)
    ax.invert_yaxis()
    ax.set_xlabel("Increase in MAE when the feature is shuffled (minutes)")
    return _finish(ax, "What the model relies on")


def _finish(ax, title):
    ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure
