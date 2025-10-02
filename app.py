# app.py — CS: EDA & Forecast (PII redacted, no captions)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from pathlib import Path
import re

st.set_page_config(page_title="CS – EDA & Forecast", layout="wide")
st.title("CS – EDA & Forecast Dashboard")

BASE = Path(__file__).parent

# ---------------------- BRAND THEME --------------------------
PRIMARY = "#268b8b"   # teal
ACCENT  = "#f3c417"   # sunflower
PURPLE  = "#87189D"   # purple
TEXT    = "#1C1C1C"
LIGHTBG = "#FFFFFF"
SECBG   = "#F8F8F8"

# Plotly template (consistent colors)
pio.templates["hub"] = pio.templates["plotly_white"]
pio.templates["hub"].layout.update(
    colorway=[PRIMARY, ACCENT, PURPLE, "#8D99AE", "#6c757d"],
    paper_bgcolor=LIGHTBG,
    plot_bgcolor=LIGHTBG,
    font=dict(color=TEXT, family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial")
)
pio.templates.default = "hub"

# Light CSS polish
st.markdown(f"""
<style>
.block-container {{max-width: 1180px; padding-top: 1rem; padding-bottom: 2rem;}}
h1, h2, h3 {{letter-spacing:.2px;}}
div[data-testid="stMetric"] {{
  border: 1px solid #eaeaea; background:{SECBG}; padding:14px; border-radius:14px;
}}
.stButton>button {{
  border-radius:10px; padding:8px 14px; font-weight:600; background:{PRIMARY}; color:white; border:0;
}}
.stButton>button:hover {{ filter: brightness(0.95); }}
hr {{margin: 1rem 0 1.2rem 0;}}
</style>
""", unsafe_allow_html=True)

# ----------------- helpers -----------------
def existing_path(candidates):
    for name in candidates:
        p = BASE / name
        if p.exists():
            return p
    return None

def read_csv_candidates(cands):
    p = existing_path(cands)
    if p is None:
        st.error(f"Missing file. Tried: {', '.join(cands)}")
        st.stop()
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Could not read {p.name}: {e}")
        st.stop()

def coerce_sem_date(df: pd.DataFrame, possible_cols=("sem_date", "ds", "date", "semester")) -> pd.DataFrame:
    col = next((c for c in possible_cols if c in df.columns), None)
    if col is None:
        st.error(f"No date/semester column found in {list(df.columns)}; expected one of {possible_cols}.")
        st.stop()
    if col != "sem_date":
        df = df.rename(columns={col: "sem_date"})
    df["sem_date"] = pd.to_datetime(df["sem_date"], errors="coerce")
    if df["sem_date"].isna().all():
        st.error("Could not parse any 'sem_date' values as dates.")
        st.stop()
    return df

def ensure_columns(df: pd.DataFrame, required: dict) -> pd.DataFrame:
    rename_map = {}
    for want, aliases in required.items():
        found = next((c for c in [want] + aliases if c in df.columns), None)
        if not found:
            st.error(f"Missing column '{want}' (aliases tried: {aliases}). Columns: {list(df.columns)}")
            st.stop()
        if found != want:
            rename_map[found] = want
    return df.rename(columns=rename_map) if rename_map else df

def detect_country_col(df: pd.DataFrame) -> str:
    for c in ["COR", "Country", "Country of Residence", "country", "Country_of_Residence"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if c != "sem_date" and df[c].dtype == "object":
            return c
    st.error("Could not detect the country column in COR forecast file.")
    st.stop()

def std_country(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    mapping = {
        "LEBANON": "Lebanon",
        "UAE": "UAE", "UNITED ARAB EMIRATES": "UAE", "EMIRATES": "UAE",
        "KSA": "Saudi Arabia", "SAUDI ARABIA": "Saudi Arabia",
        "QATAR": "Qatar", "KUWAIT": "Kuwait",
        "USA": "USA", "UNITED STATES": "USA",
    }
    return s.map(mapping).fillna(s.str.title())

# Month adjustment for display: Fall→Aug 1, Spring→Jan 1 (Summer unchanged)
def adjust_sem_month(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt): return dt
    y, m = dt.year, dt.month
    if m == 10:  # Fall encoded as Oct -> show Aug
        return pd.Timestamp(y, 8, 1)
    if m == 3:   # Spring encoded as Mar -> show Jan
        return pd.Timestamp(y, 1, 1)
    return dt

def apply_sem_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sem_date"] = df["sem_date"].map(adjust_sem_month)
    return df

# ---------- load artifacts (CSVs) ----------
actual = read_csv_candidates(["actual_enrollments.csv", "actual_enrollments (1).csv"])
fc     = read_csv_candidates(["forecast_prophet.csv", "forecast_prophet (1).csv",
                              "forecast_linear.csv", "forecast_linear (1).csv"])
cor    = read_csv_candidates(["forecast_cor.csv", "forecast_cor (1).csv"])

# standardize
actual = coerce_sem_date(actual)
actual = ensure_columns(actual, {"enrollments": ["count", "total", "Enrollments"]})
actual = apply_sem_adjustment(actual).sort_values("sem_date")[["sem_date","enrollments"]]

fc = coerce_sem_date(fc)
if "yhat" not in fc.columns and "pred_total" in fc.columns:
    fc = fc.rename(columns={"pred_total": "yhat"})
if "yhat" not in fc.columns and "pred_linear" in fc.columns:
    fc = fc.rename(columns={"pred_linear": "yhat"})
if "yhat" not in fc.columns:
    st.error(f"Forecast file missing 'yhat'. Columns: {list(fc.columns)}")
    st.stop()
if "yhat_lower" not in fc.columns: fc["yhat_lower"] = np.nan
if "yhat_upper" not in fc.columns: fc["yhat_upper"] = np.nan
fc = apply_sem_adjustment(fc).sort_values("sem_date")[["sem_date","yhat","yhat_lower","yhat_upper"]]

cor = coerce_sem_date(cor)
country_col = detect_country_col(cor)
if "pred_count" not in cor.columns:
    if "pred_total" in cor.columns and ("prop_smooth" in cor.columns or "prop" in cor.columns):
        prop_col = "prop_smooth" if "prop_smooth" in cor.columns else "prop"
        cor["pred_count"] = (cor["pred_total"] * cor[prop_col]).round().astype("Int64")
    else:
        st.error("COR CSV missing 'pred_count' (and cannot compute from 'pred_total' * 'prop').")
        st.stop()
cor["Country"] = std_country(cor[country_col])
cor = apply_sem_adjustment(cor)
cor["pred_count"] = pd.to_numeric(cor["pred_count"], errors="coerce").fillna(0).astype(int)
cor = (cor.groupby(["sem_date","Country"], as_index=False)["pred_count"]
         .sum()
         .sort_values(["sem_date","pred_count"], ascending=[True, False]))

# ---------- raw excel for EDA ----------
excel_candidates = [
    "CS - All Enrolled.xlsx",
    "CS– All Enrolled.xlsx",
    "CS- All Enrolled.xlsx",
    "CS All Enrolled.xlsx",
]
def load_raw_excel():
    p = existing_path(excel_candidates)
    if p is None:
        return pd.DataFrame()
    try:
        try:
            return pd.read_excel(p, sheet_name="All Enrolled ")
        except Exception:
            return pd.read_excel(p, sheet_name="All Enrolled")
    except Exception:
        return pd.DataFrame()

raw_df = load_raw_excel()
true_total = int(len(raw_df)) if not raw_df.empty else None

# ---- PII redaction for preview ----
def redact_pii(df: pd.DataFrame):
    drop_cols = []
    for c in df.columns:
        lc = c.lower()
        # target name/email/phone variants; don't confuse with 'Country of Residence'
        if ("full" in lc and "name" in lc) or re.fullmatch(r"\s*name\s*", lc):
            drop_cols.append(c); continue
        if any(k in lc for k in ["email", "phone", "mobile", "whatsapp"]):
            drop_cols.append(c)
    return df.drop(columns=drop_cols, errors="ignore")

# =========================
# Tabs
# =========================
tab_eda, tab_enroll, tab_cor = st.tabs(["EDA", "Enrollments Forecast", "COR Forecast"])

# -------- TAB 1: EDA (snapshot + total actual enrollments curve only) --------
with tab_eda:
    st.subheader("Exploratory Data Analysis (CS)")

    # 1) Snapshot table (PII redacted)
    if raw_df.empty:
        st.warning("Raw Excel not found in repo root. Upload the CS workbook to enable EDA.")
    else:
        st.dataframe(redact_pii(raw_df.copy()).head(50), use_container_width=True)

        # 2) KPIs (rows/cols only)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(raw_df):,}")
        c2.metric("Columns", f"{raw_df.shape[1]:,}")

    st.divider()

    # 3) Total actual enrollments over time (single teal line)
    st.subheader("Total enrollments over time")
    if actual is not None and not actual.empty:
        dfa = actual.sort_values("sem_date").copy()
        fig2 = px.line(dfa, x="sem_date", y="enrollments", markers=True, title=None)
        fig2.update_layout(
            xaxis_title="Semester",
            yaxis_title="Enrollments",
            legend_title=None,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("`actual_enrollments.csv` not found or empty — cannot draw the timeline.")

# -------- TAB 2: Enrollments Forecast --------
with tab_enroll:
    st.metric("Actual Total Enrollments", int(true_total) if true_total else int(actual["enrollments"].sum()))

    plot_df = pd.concat([
        actual.rename(columns={"enrollments":"value"}).assign(kind="Actual")[["sem_date","value","kind"]],
        fc.rename(columns={"yhat":"value"})[["sem_date","value"]].assign(kind="Forecast")
    ], ignore_index=True)

    all_sems = sorted(plot_df["sem_date"].unique())
    if all_sems:
        default_range = (all_sems[0], all_sems[-1])
        sem_range = st.select_slider(
            "Show range",
            options=all_sems,
            value=default_range,
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
        )
        mask = (plot_df["sem_date"] >= sem_range[0]) & (plot_df["sem_date"] <= sem_range[1])
        fig = px.line(plot_df[mask], x="sem_date", y="value", color="kind", markers=True,
                      title="Actual vs. Forecasted Enrollments")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for plotting.")

    # (fixed newline / scope)
    last_actual = actual["sem_date"].max() if not actual.empty else None
    future_only = fc[fc["sem_date"] > last_actual] if last_actual is not None else fc.copy()

    if not future_only.empty:
        target_sem = st.selectbox(
            "Target semester",
            list(future_only["sem_date"]),
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
        )
        target_val = int(future_only.loc[future_only["sem_date"] == target_sem, "yhat"].iloc[0])
        st.metric(f"Predicted Enrollments – {pd.to_datetime(target_sem).strftime('%b %Y')}", target_val)

        st.subheader("Next 4 Semesters")
        next4 = future_only.head(4).copy()
        next4_disp = (next4.assign(Semester=next4["sem_date"].dt.strftime("%b %Y"))
                             [["Semester","yhat"]]
                             .rename(columns={"yhat":"Predicted Enrollments"}))
        st.dataframe(next4_disp, use_container_width=True, hide_index=True)
    else:
        st.info("No future forecast rows found.")

# -------- TAB 3: COR Forecast (no caption) --------
with tab_cor:
    future_sems = sorted(cor["sem_date"].unique())
    if future_sems:
        sem_sel = st.selectbox(
            "Select a future semester",
            future_sems,
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
        )
        cor_sub = cor[cor["sem_date"] == sem_sel].sort_values("pred_count", ascending=False)

        if cor_sub.empty:
            st.info("No COR data for the selected semester.")
        else:
            fig2 = px.bar(
                cor_sub, x="pred_count", y="Country", orientation="h",
                title=f"Future COR Breakdown – {pd.to_datetime(sem_sel).strftime('%b %Y')}"
            )
            st.plotly_chart(fig2, use_container_width=True)

            tbl = (cor_sub.assign(Semester=pd.to_datetime(sem_sel).strftime("%b %Y"))
                            [["Semester","Country","pred_count"]]
                            .rename(columns={"pred_count":"Predicted Enrollments"}))
            st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.info("No COR forecast data found.")
