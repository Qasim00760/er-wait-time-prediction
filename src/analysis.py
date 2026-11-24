"""Data loading, analysis and chart functions shared by the notebook and the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "er_visits.csv"

TARGET = "Total Wait Time (min)"
STAGES = ["Time to Registration (min)", "Time to Triage (min)", "Time to Medical Professional (min)"]
URGENCY_ORDER = ["Critical", "High", "Medium", "Low"]
TIME_OF_DAY_ORDER = ["Early Morning", "Late Morning", "Afternoon", "Evening", "Night"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASON_ORDER = ["Spring", "Summer", "Fall", "Winter"]
OUTCOME_ORDER = ["Admitted", "Discharged", "Left Without Being Seen"]
URGENCY_COLORS = {"Critical": "#b2182b", "High": "#ef8a62", "Medium": "#67a9cf", "Low": "#2166ac"}
NAVY, BLUE, RED = "#0f2a44", "#3d6d99", "#b2182b"

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------- loading


def load_visits(path=DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Visit Date"])
    df["Hour"] = df["Visit Date"].dt.hour
    df["Month"] = df["Visit Date"].dt.month
    for col, order in [("Urgency Level", URGENCY_ORDER), ("Time of Day", TIME_OF_DAY_ORDER),
                       ("Day of Week", DAY_ORDER), ("Season", SEASON_ORDER), ("Patient Outcome", OUTCOME_ORDER)]:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


# ---------------------------------------------------------------- data checks


def quality_checks(df: pd.DataFrame) -> pd.Series:
    stage_sum = df[STAGES].sum(axis=1)
    hour_band = pd.cut(df["Hour"], [-1, 5, 11, 16, 20, 23],
                       labels=["Night/Early", "Morning", "Afternoon", "Evening", "Late"])
    return pd.Series(
        {
            "rows": len(df),
            "missing values": int(df.isna().sum().sum()),
            "duplicate visit IDs": int(df["Visit ID"].duplicated().sum()),
            "total wait equals sum of the 3 stages (%)": round((stage_sum == df[TARGET]).mean() * 100, 1),
            "visits whose Time of Day label matches the clock hour (%)": round(_label_matches_hour(df) * 100, 1),
            "date range": f"{df['Visit Date'].min().date()} to {df['Visit Date'].max().date()}",
        }
    )


def _label_matches_hour(df: pd.DataFrame) -> float:
    ranges = {"Early Morning": range(5, 9), "Late Morning": range(9, 12), "Afternoon": range(12, 17),
              "Evening": range(17, 21), "Night": list(range(21, 24)) + list(range(0, 5))}
    return df.apply(lambda r: r["Hour"] in ranges[r["Time of Day"]], axis=1).mean()


# ---------------------------------------------------------------- summaries


def wait_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[TARGET]
    out = pd.DataFrame(
        {
            "visits": g.size(),
            "mean_wait": g.mean().round(1),
            "median_wait": g.median(),
            "p90_wait": g.quantile(0.9).round(0),
        }
    )
    return out.reset_index()


def stage_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Average minutes spent in each stage, by urgency level."""
    out = df.groupby("Urgency Level", observed=True)[STAGES].mean().round(1)
    out.columns = ["Registration", "Triage", "Seen by clinician"]
    out["Total"] = out.sum(axis=1)
    return out.reset_index()


def outcome_by_urgency(df: pd.DataFrame) -> pd.DataFrame:
    return (pd.crosstab(df["Urgency Level"], df["Patient Outcome"], normalize="index") * 100).round(1)


def staffing_within_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """Average wait for each staffing level inside each urgency level."""
    return df.pivot_table(index="Urgency Level", columns="Nurse-to-Patient Ratio", values=TARGET,
                          aggfunc="mean", observed=True).round(1)


def correlations(df: pd.DataFrame) -> pd.Series:
    cols = ["Nurse-to-Patient Ratio", "Specialist Availability", "Facility Size (Beds)", "Hour", "Patient Satisfaction"]
    return df[cols + [TARGET]].corr()[TARGET].drop(TARGET).round(3)


# ---------------------------------------------------------------- charts


def _finish(ax, title):
    ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure


def plot_wait_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.histplot(df[TARGET], bins=50, color=BLUE, ax=ax)
    ax.axvline(df[TARGET].mean(), color=RED, linewidth=2, label=f"Mean {df[TARGET].mean():.0f} min")
    ax.axvline(df[TARGET].median(), color=NAVY, linewidth=2, linestyle="--", label=f"Median {df[TARGET].median():.0f} min")
    ax.set_xlabel("Total wait time (minutes)")
    ax.set_ylabel("Visits")
    ax.legend(frameon=False)
    return _finish(ax, "Distribution of total wait time")


def plot_wait_by_urgency(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="Urgency Level", y=TARGET, order=URGENCY_ORDER, hue="Urgency Level",
                palette=URGENCY_COLORS, legend=False, fliersize=2, width=0.55, ax=ax)
    ax.set_xlabel("Urgency level at triage")
    ax.set_ylabel("Total wait (minutes)")
    return _finish(ax, "Wait time by urgency level")


def plot_stage_breakdown(df: pd.DataFrame):
    s = stage_breakdown(df).set_index("Urgency Level")[["Registration", "Triage", "Seen by clinician"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = pd.Series(0.0, index=s.index)
    for col, color in zip(s.columns, ["#9ecae1", "#3d6d99", NAVY]):
        ax.barh(s.index.astype(str), s[col], left=left, color=color, label=col)
        left += s[col]
    for i, total in enumerate(left):
        ax.text(total + 2, i, f"{total:.0f} min", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Average minutes")
    ax.set_xlim(0, left.max() * 1.15)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(ax, "Where the wait happens: average minutes per stage")


def plot_wait_by_staffing(df: pd.DataFrame):
    s = wait_summary(df, "Nurse-to-Patient Ratio")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(s["Nurse-to-Patient Ratio"].astype(str), s["mean_wait"], color=BLUE)
    for i, (v, n) in enumerate(zip(s["mean_wait"], s["visits"])):
        ax.text(i, v + 3, f"{v:.0f} min\n(n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, s["mean_wait"].max() * 1.25)
    ax.set_xlabel("Nurse-to-patient ratio (patients per nurse)")
    ax.set_ylabel("Average wait (minutes)")
    return _finish(ax, "Average wait by staffing level (mostly reflects urgency)")


def plot_wait_by_time_of_day(df: pd.DataFrame):
    pivot = df.pivot_table(index="Time of Day", columns="Urgency Level", values=TARGET, aggfunc="mean", observed=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", linewidths=1,
                cbar_kws={"label": "Average wait (min)"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Average wait by time of day and urgency", fontsize=13, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_arrivals_by_hour(df: pd.DataFrame):
    counts = df["Hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(counts.index, counts.values, color=BLUE)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of arrival")
    ax.set_ylabel("Visits")
    return _finish(ax, "Arrivals by hour of day")


def plot_outcome_by_urgency(df: pd.DataFrame):
    t = outcome_by_urgency(df)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = pd.Series(0.0, index=t.index)
    for col, color in zip(OUTCOME_ORDER, [NAVY, "#9ecae1", RED]):
        ax.barh(t.index.astype(str), t[col], left=left, color=color, label=col)
        left += t[col]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of visits (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return _finish(ax, "Patient outcome by urgency level")
