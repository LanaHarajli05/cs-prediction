# app.py — CS: EDA + Forecast Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="CS – EDA & Forecast", layout="wide")
st.title("CS – EDA & Forecast Dashboard")

BASE = Path(__file__).parent

# =========================
# Helpers (files & columns)
# =========================
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
    out = s.map(mapping).fillna(s.str.title())
    return out

# Semester month adjustment: Fall→Aug 1, Spring→Jan 1 (Summer unchanged)
def adjust_sem_month(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt): return dt
    y, m = dt.year, dt.month
    if m == 10:  # Fall previously encoded as Oct
        return pd.Timestamp(y, 8, 1)
    if m == 3:   # Spring previously encoded as Mar
        return pd.Timestamp(y, 1, 1)
    return dt

def apply_sem_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sem_date"] = df["sem_date"].map(adjust_sem_month)
    return df

# =========================
# Load artifacts (CSVs)
# =========================
actual = read_csv_candidates(["actual_enrollments.csv", "actual_enrollments (1).csv"])
fc     = read_csv_candidates(["forecast_prophet.csv", "forecast_prophet (1).csv",
                              "forecast_linear.csv", "forecast_linear (1).csv"])
cor    = read_csv_candidates(["forecast_cor.csv", "forecast_cor (1).csv"])

# Standardize
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
cor = (cor.groupby(["sem_date","Country"], as_index=False)["pred_count"]
         .sum()
         .sort_values(["sem_date","pred_count"], ascending=[True, False]))

# =========================
# Load raw Excel for EDA & true total
# =========================
excel_candidates = [
    "CS - All Enrolled.xlsx",
    "CS– All Enrolled.xlsx",
    "CS- All Enrolled.xlsx",
    "CS All Enrolled.xlsx",
]
excel_path = existing_path(excel_candidates)

@st.cache_data(show_spinner=False)
def load_raw_excel(p: Path) -> pd.DataFrame:
    if p is None:
        return pd.DataFrame()
    try:
        # workbook sheet has a known trailing space variant
        try:
            df0 = pd.read_excel(p, sheet_name="All Enrolled ")
        except Exception:
            df0 = pd.read_excel(p, sheet_name="All Enrolled")
        return df0
    except Exception:
        return pd.DataFrame()

raw_df = load_raw_excel(excel_path)
true_total = int(len(raw_df)) if not raw_df.empty else None

# =========================
# Tabs: EDA, Enrollments, COR
# =========================
tab_eda, tab_enroll, tab_cor = st.tabs(["EDA", "Enrollments Forecast", "COR Forecast"])

# -------------------------
# TAB 1: EDA
# -------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis (CS)")
    if raw_df.empty:
        st.warning("Raw Excel not found in repo root. Upload the CS workbook to enable EDA.")
    else:
        # Basic overview
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(raw_df):,}")
        c2.metric("Columns", f"{raw_df.shape[1]:,}")
        c3.metric("Missing cells", f"{int(raw_df.isna().sum().sum()):,}")
        dup_guess_col = next((c for c in ["Email Address", "Email", "NAME", "Full Name"] if c in raw_df.columns), None)
        dups = raw_df.duplicated(subset=[dup_guess_col]).sum() if dup_guess_col else raw_df.duplicated().sum()
        c4.metric("Duplicate rows", f"{int(dups):,}")

        with st.expander("Columns & Types"):
            info_df = pd.DataFrame({
                "column": raw_df.columns,
                "dtype": [str(t) for t in raw_df.dtypes.values],
                "missing_%": (raw_df.isna().mean() * 100).round(1),
                "unique": [raw_df[c].nunique(dropna=True) for c in raw_df.columns]
            }).sort_values("missing_%", ascending=False)
            st.dataframe(info_df, use_container_width=True)

        # Guess demographics-like columns
        demo_candidates = {
            "Gender": [c for c in raw_df.columns if "gender" in c.lower()],
            "Age": [c for c in raw_df.columns if c.lower() in ["age","age "] or "age(" in c.lower()],
            "Employment": [c for c in raw_df.columns if any(k in c.lower() for k in ["employment", "job", "work", "occupation"])],
            "Education": [c for c in raw_df.columns if any(k in c.lower() for k in ["education", "degree", "major"])],
            "Country of Residence": [c for c in raw_df.columns if "country" in c.lower() or "cor" in c.lower()],
            "Cohort": [c for c in raw_df.columns if "cohort" in c.lower()],
        }

        st.markdown("### Demographics at a Glance")
        demo_cols = st.multiselect(
            "Pick demographic columns to summarize",
            sorted(set(sum(demo_candidates.values(), []))),
            default=[x for x in sum(demo_candidates.values(), []) if x][:4]
        )

        if demo_cols:
            for col in demo_cols:
                vc = (raw_df[col].astype(str).str.strip()
                      .replace({"nan": np.nan}).dropna()).value_counts().head(15)
                fig = px.bar(vc[::-1], orientation="h", title=f"Top values — {col}")
                st.plotly_chart(fig, use_container_width=True)

        # Cohort timeline (map to Aug/Jan/Jul for Fall/Spring/Summer if present)
        def clean_cohort_text(s):
            if pd.isna(s): return s
            s = str(s).strip()
            s = s.replace(" -", "-").replace("- ", "-")
            return s
        def cohort_to_date(cohort):
            # Fall YY-YY -> Aug 1 of first year, Spring -> Jan 1 of second year, Summer -> Jul 1
            import re
            if pd.isna(cohort): return pd.NaT
            txt = str(cohort).strip()
            m = re.match(r'^(Fall|Spring|Summer)\s+(\d{2})-(\d{2})$', txt)
            if not m: return pd.NaT
            season, y1, y2 = m.groups()
            y1, y2 = 2000 + int(y1), 2000 + int(y2)
            if season == "Fall":   return pd.Timestamp(y1, 8, 1)
            if season == "Spring": return pd.Timestamp(y2, 1, 1)
            if season == "Summer": return pd.Timestamp(y2, 7, 1)
            return pd.NaT

        coh_col = demo_candidates["Cohort"][0] if demo_candidates["Cohort"] else None
        if coh_col:
            tmp = raw_df.copy()
            tmp["Cohort_clean"] = tmp[coh_col].map(clean_cohort_text)
            tmp["sem_date"] = tmp["Cohort_clean"].map(cohort_to_date)
            ts = (tmp.dropna(subset=["sem_date"])
                    .groupby("sem_date").size().rename("enrollments")
                    .reset_index().sort_values("sem_date"))
            if not ts.empty:
                fig_ts = px.line(ts, x="sem_date", y="enrollments",
                                 markers=True, title="Enrollments by Cohort (Aug=Fall, Jan=Spring)")
                st.plotly_chart(fig_ts, use_container_width=True)

        # Correlation matrix for numeric columns
        st.markdown("### Correlation Matrix (Numeric Columns)")
        num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            corr = raw_df[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation (Pearson)")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric columns detected for correlation.")

        # Histograms / scatter matrix
        st.markdown("### Distributions")
        sel_num = st.multiselect("Numeric columns to visualize (histograms)",
                                 num_cols, default=num_cols[: min(4, len(num_cols))])
        for col in sel_num:
            fig_h = px.histogram(raw_df, x=col, nbins=30, title=f"Histogram — {col}")
            st.plotly_chart(fig_h, use_container_width=True)

        # Missingness
        st.markdown("### Missingness by Column")
        miss = raw_df.isna().mean().sort_values(ascending=False) * 100
        miss_df = miss.round(1).rename("missing_%").reset_index(names="column")
        fig_miss = px.bar(miss_df.head(25), x="missing_%", y="column", orientation="h",
                          title="Top 25 Columns by Missing %")
        st.plotly_chart(fig_miss, use_container_width=True)

        with st.expander("Preview raw data"):
            st.dataframe(raw_df.head(50), use_container_width=True)

# -------------------------
# TAB 2: Enrollments Forecast
# -------------------------
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
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),  # Jan/Spring, Aug/Fall
        )
        mask = (plot_df["sem_date"] >= sem_range[0]) & (plot_df["sem_date"] <= sem_range[1])
        fig = px.line(plot_df[mask], x="sem_date", y="value", color="kind", markers=True,
                      title="Actual vs. Forecasted Enrollments (Jan=Spring, Aug=Fall)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for plotting.")

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

# -------------------------
# TAB 3: COR Forecast
# -------------------------
with tab_cor:
    st.caption("Forecasted enrollments by Country of Residence (deduplicated & standardized).")
    future_sems = sorted(cor["sem_date"].unique())
    if future_sems:
        sem_sel = st.selectbox(
            "Select a future semester",
            future_sems,
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
        )
        cor_sub = cor[cor["sem_date"] == sem_sel].sort_values("pred_count", ascending=False)

        fig2 = px.bar(cor_sub, x="pred_count", y="Country", orientation="h",
                      title=f"Future COR Breakdown – {pd.to_datetime(sem_sel).strftime('%b %Y')}")
        st.plotly_chart(fig2, use_container_width=True)

        tbl = (cor_sub.assign(Semester=pd.to_datetime(sem_sel).strftime("%b %Y"))
                        [["Semester","Country","pred_count"]]
                        .rename(columns={"pred_count":"Predicted Enrollments"}))
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.info("No COR forecast data found.")

st.caption("Notes: Fall is displayed in **August**, Spring in **January**. EDA reads the raw Excel file; forecasts use precomputed CSVs.")
