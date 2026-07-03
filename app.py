# -*- coding: utf-8 -*-
"""
🧠 Brainy Explorer — Universal Data Exploration Studio
--------------------------------------------------------
Originally sketched out in a Colab workshop notebook (PythonWorkshop Day 2),
rebuilt as a Streamlit app. All the exploration + modelling-hint logic from
that notebook lives here — but instead of being hard-wired to one dataset
and one target column ("sensor_quality"), everything is inferred at runtime:
column types, likely target candidates, task type (classification vs.
regression), and which charts even make sense to draw.

Upload a CSV or Excel file, pick a target, and the app takes it from there.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

sns.set_style("whitegrid")

# ============================================================
# PAGE CONFIG + THEME
# ============================================================

st.set_page_config(
    page_title="Brainy Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0F1023;
    --surface: #171A35;
    --surface-2: #1F2347;
    --accent: #8B5CF6;
    --accent-2: #22D3EE;
    --spark: #F472B6;
    --text: #E8E6F5;
    --text-dim: #A6A3C9;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #1B1E44 0%, #0F1023 45%),
                var(--bg);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Hero title with animated synapse gradient */
.brainy-hero {
    padding: 1.6rem 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #1B1E44 0%, #21254F 60%, #191C3E 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}

.brainy-hero::after {
    content: "";
    position: absolute;
    top: -60%; right: -20%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(34,211,238,0.25) 0%, rgba(34,211,238,0) 70%);
    pointer-events: none;
}

.brainy-title {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent-2), var(--accent) 55%, var(--spark));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.15rem;
}

.brainy-subtitle {
    color: var(--text-dim);
    font-size: 0.98rem;
    max-width: 620px;
}

/* Synapse divider */
.synapse-divider {
    height: 2px;
    margin: 1.6rem 0 1.2rem 0;
    background: linear-gradient(90deg, transparent, var(--accent) 15%, var(--accent-2) 50%, var(--spark) 85%, transparent);
    border-radius: 999px;
    opacity: 0.75;
}

/* Section eyebrow labels */
.eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: var(--accent-2);
    font-weight: 600;
    margin-bottom: 0.2rem;
}

/* Metric-style cards */
.brain-card {
    background: var(--surface);
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    height: 100%;
}

.brain-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
}

.brain-card .label {
    color: var(--text-dim);
    font-size: 0.82rem;
    margin-top: 0.1rem;
}

section[data-testid="stSidebar"] {
    background: #12142C;
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}

.stButton>button {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: #0F1023;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
}

.stButton>button:hover {
    filter: brightness(1.08);
    color: #0F1023;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    background-color: var(--surface);
    border-radius: 10px 10px 0 0;
    padding: 0.5rem 1rem;
    color: var(--text-dim);
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    background-color: var(--surface-2) !important;
    border-bottom: 2px solid var(--accent-2);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

BRAIN_EMOJIS = ["🧠", "✨", "🔬", "⚡", "🧬"]


def eyebrow(text: str):
    st.markdown(f"<div class='eyebrow'>{text}</div>", unsafe_allow_html=True)


def divider():
    st.markdown("<div class='synapse-divider'></div>", unsafe_allow_html=True)


def metric_card(col, value, label):
    col.markdown(
        f"<div class='brain-card'><div class='value'>{value}</div>"
        f"<div class='label'>{label}</div></div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="brainy-hero">
        <div class="brainy-title">🧠 Brainy Explorer</div>
        <div class="brainy-subtitle">
            Drop in any dataset — CSV or Excel — and this app wires up its own
            neurons: it profiles your columns, lets you pick a target on the fly,
            and figures out whether you're looking at a classification or
            regression problem before running exploration, correlations, and a
            quick KNN baseline. No hard-coded columns. Ever. ⚡
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR — UPLOAD
# ============================================================

with st.sidebar:
    st.markdown("### 🧬 Feed the brain")
    st.caption("Upload a CSV or Excel file to get started.")
    uploaded_file = st.file_uploader(
        "Upload dataset", type=["csv", "xlsx", "xls"], label_visibility="collapsed"
    )
    load_clicked = st.button("🚀 Load & analyze data", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Built from a Python workshop notebook, reborn as a Streamlit app. "
        "Handles missing values, mixed types, and auto-detects your target."
    )


# ============================================================
# DATA LOADING (cached, robust to csv/xlsx and messy encodings)
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    # Try a few common encodings/separators for CSVs before giving up
    for encoding in ("utf-8", "latin1", "utf-16"):
        try:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding=encoding, sep=None, engine="python")
        except Exception:
            continue
    buffer.seek(0)
    return pd.read_csv(buffer)


def guess_target_candidates(df: pd.DataFrame) -> list:
    """Rank columns by how 'target-like' they look, without assuming any name."""
    name_hints = ("target", "label", "class", "y", "outcome", "result",
                  "quality", "score", "status", "category", "churn", "price")
    scored = []
    n = len(df)
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1 or nunique == n:
            continue  # constant or purely an ID column — unlikely target
        score = 0
        lname = col.lower()
        if any(h in lname for h in name_hints):
            score += 5
        if df[col].dtype == object or nunique <= max(10, int(n * 0.05)):
            score += 2  # looks categorical -> plausible classification target
        if pd.api.types.is_numeric_dtype(df[col]):
            score += 1  # numeric -> plausible regression target
        if col == df.columns[-1]:
            score += 1  # last column is a common target convention
        scored.append((score, col))
    scored.sort(reverse=True)
    return [c for _, c in scored] or list(df.columns)


def infer_task_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
        return "regression"
    return "classification"


# ============================================================
# MAIN FLOW
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.filename = None

if load_clicked:
    if uploaded_file is None:
        st.sidebar.error("Please choose a file before loading. 🧠")
    else:
        with st.spinner("Waking up the neurons..."):
            try:
                df_loaded = load_dataframe(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.df = df_loaded
                st.session_state.filename = uploaded_file.name
            except Exception as e:
                st.sidebar.error(f"Couldn't read that file: {e}")

df = st.session_state.df

if df is None:
    st.info("👈 Upload a CSV or Excel file in the sidebar, then hit **Load & analyze data** to begin.")
    st.markdown(
        """
        <div class="brain-card" style="margin-top: 1rem;">
        <b>What this app does once you upload data:</b><br><br>
        🔬 Profiles every column — types, missingness, cardinality<br>
        🧬 Cleans and encodes categorical data automatically<br>
        🎯 Lets you pick <i>any</i> column as the target, with smart suggestions<br>
        📊 Runs correlation analysis and distribution exploration<br>
        ⚡ Trains a quick KNN baseline and shows you how it performs<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.success(f"{np.random.choice(BRAIN_EMOJIS)} Loaded **{st.session_state.filename}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns.")

tabs = st.tabs([
    "🔬 Overview",
    "🧬 Clean & Encode",
    "🎯 Target & Correlations",
    "📊 Distributions",
    "⚡ Modeling Hints",
])

# ------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------
with tabs[0]:
    eyebrow("Shape & structure")
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{df.shape[0]:,}", "Rows")
    metric_card(c2, f"{df.shape[1]:,}", "Columns")
    metric_card(c3, f"{df.isnull().sum().sum():,}", "Missing values")
    metric_card(c4, f"{df.duplicated().sum():,}", "Duplicate rows")

    divider()
    eyebrow("First look")
    st.dataframe(df.head(10), use_container_width=True)

    divider()
    eyebrow("Column profile")
    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
        "missing_%": (df.isnull().mean() * 100).round(2),
        "unique_values": df.nunique(),
    })
    st.dataframe(profile, use_container_width=True)

    divider()
    eyebrow("Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

# ------------------------------------------------------------
# TAB 2 — CLEAN & ENCODE
# ------------------------------------------------------------
with tabs[1]:
    eyebrow("Missing value handling")
    st.markdown(
        "Rows containing missing values are dropped for the exploration below "
        "(the same approach as the original workshop notebook), keeping the "
        "logic simple and transparent."
    )

    df_clean = df.dropna().copy()
    st.write(
        f"🧠 Dataset shape after dropping missing rows: "
        f"**{df_clean.shape[0]:,} rows × {df_clean.shape[1]:,} columns** "
        f"(was {df.shape[0]:,} rows)."
    )

    divider()
    eyebrow("Categorical encoding")
    categorical_columns = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical_columns:
        st.write(f"Encoding **{len(categorical_columns)}** categorical column(s): `{', '.join(categorical_columns)}`")
        df_encoded = df_clean.copy()
        encoders = {}
        for column in categorical_columns:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            encoders[column] = le
        st.dataframe(df_encoded.head(10), use_container_width=True)
    else:
        st.write("No categorical columns detected — nothing to encode. ✨")
        df_encoded = df_clean.copy()

    st.session_state.df_clean = df_clean
    st.session_state.df_encoded = df_encoded

# ------------------------------------------------------------
# TAB 3 — TARGET & CORRELATIONS
# ------------------------------------------------------------
with tabs[2]:
    df_encoded = st.session_state.get("df_encoded", df.dropna())
    eyebrow("Pick your target variable")
    st.markdown(
        "No target is hard-coded — the app ranks columns by how likely they "
        "are to be a target (name, cardinality, position) and lets you confirm."
    )

    candidates = guess_target_candidates(df_encoded)
    target_col = st.selectbox(
        "🎯 Target variable",
        options=candidates,
        index=0,
        help="Ranked by how 'target-like' each column looks. Pick whichever fits your task.",
    )

    task_type = infer_task_type(df_encoded[target_col])
    st.markdown(f"🧠 Inferred task type: **{task_type.title()}**  \n"
                f"(based on the target's data type and number of unique values)")

    st.session_state.target_col = target_col
    st.session_state.task_type = task_type

    divider()
    eyebrow("Correlation matrix")
    numeric_df = df_encoded.select_dtypes(include=[np.number])

    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        corr = numeric_df.corr()
        sns.heatmap(corr, cmap="coolwarm", ax=ax, center=0, linewidths=0.3)
        ax.set_title("Correlation Matrix", color="white")
        ax.tick_params(colors="white")
        plt.setp(ax.get_xticklabels(), color="white")
        plt.setp(ax.get_yticklabels(), color="white")
        st.pyplot(fig, use_container_width=True)

        if target_col in corr.columns:
            divider()
            eyebrow(f"Correlation with '{target_col}'")
            target_corr = corr[target_col].drop(target_col).sort_values(key=abs, ascending=False)
            st.dataframe(target_corr.to_frame("correlation"), use_container_width=True)
    else:
        st.info("Not enough numeric columns to build a correlation matrix.")

# ------------------------------------------------------------
# TAB 4 — DISTRIBUTIONS
# ------------------------------------------------------------
with tabs[3]:
    df_encoded = st.session_state.get("df_encoded", df.dropna())
    eyebrow("Explore a column's distribution")

    col_choice = st.selectbox("Choose a column to visualize", options=df_encoded.columns.tolist())

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    if pd.api.types.is_numeric_dtype(df_encoded[col_choice]):
        sns.histplot(df_encoded[col_choice], kde=True, ax=ax, color="#8B5CF6")
    else:
        vc = df_encoded[col_choice].astype(str).value_counts().head(20)
        sns.barplot(x=vc.values, y=vc.index, ax=ax, color="#22D3EE")

    ax.set_title(f"Distribution of {col_choice}", color="white")
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), color="white")
    plt.setp(ax.get_yticklabels(), color="white")
    st.pyplot(fig, use_container_width=True)

    divider()
    eyebrow(f"Target balance: '{st.session_state.get('target_col', '')}'")
    if "target_col" in st.session_state:
        target_col = st.session_state.target_col
        if st.session_state.task_type == "classification":
            counts = df_encoded[target_col].value_counts()
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax2, color="#F472B6")
            ax2.set_title("Class balance", color="white")
            ax2.tick_params(colors="white")
            plt.setp(ax2.get_xticklabels(), color="white")
            plt.setp(ax2.get_yticklabels(), color="white")
            st.pyplot(fig2, use_container_width=True)
        else:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")
            sns.histplot(df_encoded[target_col], kde=True, ax=ax2, color="#F472B6")
            ax2.set_title("Target distribution", color="white")
            ax2.tick_params(colors="white")
            plt.setp(ax2.get_xticklabels(), color="white")
            plt.setp(ax2.get_yticklabels(), color="white")
            st.pyplot(fig2, use_container_width=True)

# ------------------------------------------------------------
# TAB 5 — MODELING HINTS (baseline KNN, from the original notebook)
# ------------------------------------------------------------
with tabs[4]:
    df_encoded = st.session_state.get("df_encoded", None)

    if df_encoded is None or "target_col" not in st.session_state:
        st.info("Pick a target variable in the 'Target & Correlations' tab first.")
    else:
        target_col = st.session_state.target_col
        task_type = st.session_state.task_type

        eyebrow("Baseline model")
        st.markdown(
            f"Training a quick **KNN {'classifier' if task_type == 'classification' else 'regressor'}** "
            f"as a sanity-check baseline — same approach as the original workshop notebook, "
            f"just generalized to whatever target you pick."
        )

        run_model = st.button("⚡ Train baseline KNN model")

        if run_model:
            with st.spinner("Training..."):
                X = df_encoded.drop(columns=[target_col])
                y = df_encoded[target_col]
                X = X.select_dtypes(include=[np.number])  # safety net

                if X.shape[1] == 0:
                    st.error("No numeric feature columns left after removing the target. Try a different target.")
                else:
                    stratify = y if task_type == "classification" and y.nunique() > 1 else None
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.20, random_state=42, stratify=stratify
                        )
                    except ValueError:
                        # Falls back to unstratified split when a class has too few members
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.20, random_state=42
                        )

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    n_neighbors = min(21, max(1, len(X_train) - 1))

                    if task_type == "classification":
                        model = KNeighborsClassifier(n_neighbors=n_neighbors)
                        model.fit(X_train_s, y_train)
                        y_pred = model.predict(X_test_s)

                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                        c1, c2, c3, c4 = st.columns(4)
                        metric_card(c1, f"{accuracy:.3f}", "Accuracy")
                        metric_card(c2, f"{precision:.3f}", "Precision")
                        metric_card(c3, f"{recall:.3f}", "Recall")
                        metric_card(c4, f"{f1:.3f}", "F1-score")

                        divider()
                        eyebrow("Classification report")
                        st.text(classification_report(y_test, y_pred, zero_division=0))

                        divider()
                        eyebrow("Confusion matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        fig.patch.set_alpha(0)
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_xlabel("Predicted", color="white")
                        ax.set_ylabel("Actual", color="white")
                        ax.tick_params(colors="white")
                        st.pyplot(fig, use_container_width=True)

                    else:
                        model = KNeighborsRegressor(n_neighbors=n_neighbors)
                        model.fit(X_train_s, y_train)
                        y_pred = model.predict(X_test_s)

                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)

                        c1, c2, c3 = st.columns(3)
                        metric_card(c1, f"{mae:.3f}", "MAE")
                        metric_card(c2, f"{rmse:.3f}", "RMSE")
                        metric_card(c3, f"{r2:.3f}", "R²")

                        divider()
                        eyebrow("Predicted vs. actual")
                        fig, ax = plt.subplots(figsize=(7, 6))
                        fig.patch.set_alpha(0)
                        ax.set_facecolor("none")
                        ax.scatter(y_test, y_pred, alpha=0.6, color="#22D3EE")
                        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                        ax.plot(lims, lims, color="#F472B6", linestyle="--")
                        ax.set_xlabel("Actual", color="white")
                        ax.set_ylabel("Predicted", color="white")
                        ax.tick_params(colors="white")
                        st.pyplot(fig, use_container_width=True)

                    divider()
                    eyebrow("Learning curve")
                    try:
                        cv_folds = min(5, max(2, y.value_counts().min() if task_type == "classification" else 5))
                        train_sizes, train_scores, val_scores = learning_curve(
                            estimator=model,
                            X=scaler.transform(X),
                            y=y,
                            cv=cv_folds,
                            scoring="accuracy" if task_type == "classification" else "r2",
                            train_sizes=np.linspace(0.2, 1.0, 6),
                            n_jobs=-1,
                        )
                        train_mean = train_scores.mean(axis=1)
                        val_mean = val_scores.mean(axis=1)

                        fig, ax = plt.subplots(figsize=(9, 5))
                        fig.patch.set_alpha(0)
                        ax.set_facecolor("none")
                        ax.plot(train_sizes, train_mean, marker="o", label="Training", color="#8B5CF6")
                        ax.plot(train_sizes, val_mean, marker="s", label="Cross-validation", color="#22D3EE")
                        ax.legend(facecolor="#171A35", labelcolor="white")
                        ax.set_xlabel("Training examples", color="white")
                        ax.set_ylabel("Score", color="white")
                        ax.tick_params(colors="white")
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Learning curve skipped ({e}) — usually happens with very small or imbalanced datasets.")

                    st.success("🧠 Baseline complete. This is a sanity check, not a final model — use it to gauge whether the signal is there before investing in tuning.")

st.markdown("<div class='synapse-divider'></div>", unsafe_allow_html=True)
st.caption("🧠 Brainy Explorer — built for exploring whatever dataset you throw at it.")
