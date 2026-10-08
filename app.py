import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CREDITWISE — LOAN APPROVAL PREDICTION APP
# Complete replacement app.py
# ============================================================

st.set_page_config(
    page_title="CreditWise",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# MODEL / DATA CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# These are the exact 18 raw columns used by the corrected
# deployment pipeline.
NUMERICAL_COLUMNS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]

CATEGORICAL_COLUMNS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]

MODEL_INPUT_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

CATEGORY_OPTIONS = {
    "Employment_Status": [
        "Salaried",
        "Self-Employed",
        "Business",
    ],
    "Marital_Status": [
        "Single",
        "Married",
    ],
    "Loan_Purpose": [
        "Home",
        "Education",
        "Personal",
        "Business",
    ],
    "Property_Area": [
        "Urban",
        "Semi-Urban",
        "Rural",
    ],
    "Education_Level": [
        "Graduate",
        "Postgraduate",
        "Undergraduate",
    ],
    "Gender": [
        "Male",
        "Female",
    ],
    "Employer_Category": [
        "Govt",
        "Private",
        "Self",
    ],
}


# ============================================================
# MODEL LOADING
# ============================================================

def _make_one_hot_encoder():
    """Create an encoder compatible with the installed scikit-learn version."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse=False,
        )


def train_compatible_model():
    """
    Rebuild the deployment pipeline with the SAME scikit-learn version
    currently running the Streamlit app.

    The previous model.pkl was serialized with a different scikit-learn
    version. That is why SimpleImputer can load but later raises:
        'SimpleImputer' object has no attribute '_fill_dtype'

    Re-training here removes that binary/pickle compatibility problem.
    """
    data_path = BASE_DIR / "loan_approval_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "loan_approval_data.csv was not found. It is required to rebuild "
            "model.pkl with the current scikit-learn version."
        )

    df = pd.read_csv(data_path)

    if "Applicant_ID" in df.columns:
        df = df.drop(columns=["Applicant_ID"])

    if "Loan_Approved" not in df.columns:
        raise ValueError("The dataset does not contain the Loan_Approved target column.")

    df = df.dropna(subset=["Loan_Approved"]).copy()

    # The original CSV stores the target as strings: "Yes" / "No".
    # Do not rely on pandas dtype == object here because pandas can load the
    # column as StringDtype or another extension dtype. Normalize the values
    # explicitly, then convert the cleaned target to integer 0/1.
    target = df["Loan_Approved"].astype("string").str.strip().str.lower()
    target_map = {
        "no": 0,
        "yes": 1,
        "0": 0,
        "1": 1,
    }

    # First handle the known Yes/No/0/1 representations.
    mapped_target = target.map(target_map)

    # If the dataset contains any other numeric-looking values, accept them
    # as numeric as well. Unknown text values remain NaN and are removed below.
    numeric_target = pd.to_numeric(target, errors="coerce")
    df["Loan_Approved"] = mapped_target.fillna(numeric_target)

    df = df.dropna(subset=["Loan_Approved"]).copy()

    # Validate the final target before converting it. This prevents errors such
    # as: invalid literal for int() with base 10: 'No'
    unique_targets = set(pd.to_numeric(df["Loan_Approved"], errors="coerce").dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            "Loan_Approved must contain only Yes/No or 0/1 values. "
            f"Found: {sorted(unique_targets)}"
        )

    df["Loan_Approved"] = pd.to_numeric(
        df["Loan_Approved"], errors="raise"
    ).astype(int)

    missing = [column for column in MODEL_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required model columns: " + ", ".join(missing)
        )

    X = df[MODEL_INPUT_COLUMNS].copy()
    y = df["Loan_Approved"].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

    compatible_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    compatible_model.fit(x_train, y_train)

    # Save a fresh pickle produced by the currently installed sklearn.
    joblib.dump(compatible_model, MODEL_PATH)

    return compatible_model


def _model_has_old_imputer_state(model_object):
    """
    Detect the specific SimpleImputer compatibility problem before prediction.
    """
    try:
        params = model_object.get_params(deep=True)
    except Exception:
        return False

    for value in params.values():
        if isinstance(value, SimpleImputer) and not hasattr(value, "_fill_dtype"):
            return True

    return False


@st.cache_resource
def load_model():
    """Load model.pkl, rebuilding it automatically if its sklearn pickle state is incompatible."""
    if not MODEL_PATH.exists():
        return train_compatible_model()

    try:
        loaded_model = joblib.load(MODEL_PATH)
    except Exception:
        # A completely unreadable/incompatible pickle is also rebuilt from the
        # original training CSV rather than leaving the application unusable.
        return train_compatible_model()

    if _model_has_old_imputer_state(loaded_model):
        return train_compatible_model()

    return loaded_model


try:
    model = load_model()
except Exception as exc:
    st.error("CreditWise could not load or rebuild the trained model.")
    st.code(str(exc))
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def get_model_expected_columns(model_object):
    """
    Read the feature names stored by the fitted sklearn pipeline.

    The corrected model is a Pipeline containing:
        ColumnTransformer -> LogisticRegression

    feature_names_in_ should contain the original 18 raw columns.
    """
    expected = getattr(model_object, "feature_names_in_", None)

    if expected is None:
        return MODEL_INPUT_COLUMNS.copy()

    return [str(column) for column in list(expected)]


def prepare_model_input(values):
    """
    Create a DataFrame using the exact raw feature names.

    IMPORTANT:
    We do NOT manually scale, encode, impute, or engineer features here.
    model.pkl already contains the preprocessing pipeline.
    """
    input_df = pd.DataFrame([values])

    # Force the exact model column order.
    expected_columns = get_model_expected_columns(model)

    missing_columns = [
        column for column in expected_columns
        if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following model features are missing: "
            + ", ".join(missing_columns)
        )

    # This fixes the common "feature mismatch" problem caused by
    # DataFrame column order / strict column comparison.
    input_df = input_df.reindex(columns=expected_columns)

    return input_df


def predict_loan(values):
    """
    Predict using the fitted sklearn pipeline.

    If an old cached model somehow reaches this point with the legacy
    SimpleImputer state, rebuild the pipeline once and retry automatically.
    """
    global model

    input_df = prepare_model_input(values)

    try:
        prediction = int(model.predict(input_df)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

    except AttributeError as exc:
        if "_fill_dtype" not in str(exc):
            raise

        # Clear the cached resource so the freshly trained pipeline becomes
        # the active model for this Streamlit process.
        load_model.clear()
        model = train_compatible_model()

        prediction = int(model.predict(input_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

    return prediction, probability, input_df


def reset_result():
    st.session_state["prediction"] = None
    st.session_state["probability"] = None


# ============================================================
# SESSION STATE
# ============================================================

if "intro_seen" not in st.session_state:
    st.session_state["intro_seen"] = False

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

if "probability" not in st.session_state:
    st.session_state["probability"] = None

if "last_input" not in st.session_state:
    st.session_state["last_input"] = None

if "show_processing" not in st.session_state:
    st.session_state["show_processing"] = False


# ============================================================
# GLOBAL PREMIUM CSS
# ============================================================

GLOBAL_CSS = r"""
<style>

:root {
    --cw-bg: #050609;
    --cw-panel: #0b0d12;
    --cw-panel-2: #10131a;
    --cw-panel-3: #151923;
    --cw-border: rgba(255,255,255,.11);
    --cw-border-strong: rgba(255,255,255,.18);
    --cw-text: #f4f5f7;
    --cw-muted: #9da3af;
    --cw-soft: #c9ced7;
    --cw-white: #ffffff;
    --cw-success: #7cf7bd;
    --cw-danger: #ff7b86;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 50% -10%, rgba(255,255,255,.07), transparent 34%),
        radial-gradient(circle at 10% 30%, rgba(255,255,255,.025), transparent 30%),
        #050609 !important;
    color: var(--cw-text) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    visibility: hidden !important;
}

[data-testid="stSidebar"] {
    background: #080a0e !important;
    border-right: 1px solid rgba(255,255,255,.08) !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1440px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

.block-container {
    max-width: 1440px !important;
}

h1, h2, h3, h4, h5, h6,
p, label, span, div {
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.cw-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding: 18px 0 24px;
}

.cw-brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.cw-brand-mark {
    width: 42px;
    height: 42px;
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 13px;
    display: grid;
    place-items: center;
    background:
        linear-gradient(145deg, rgba(255,255,255,.14), rgba(255,255,255,.025));
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.1),
        0 12px 40px rgba(0,0,0,.35);
}

.cw-brand-name {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -.03em;
}

.cw-brand-sub {
    color: var(--cw-muted);
    font-size: .76rem;
    margin-top: 2px;
}

.cw-kicker {
    color: #9da3af;
    font-size: .72rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.cw-hero-title {
    font-size: clamp(2.8rem, 6vw, 6rem);
    line-height: .9;
    font-weight: 900;
    letter-spacing: -.075em;
    margin: 0;
    background:
        linear-gradient(180deg, #ffffff 0%, #d5d8df 46%, #777d88 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.cw-hero-copy {
    max-width: 740px;
    color: #aeb4bf;
    font-size: 1rem;
    line-height: 1.75;
    margin-top: 20px;
}

.cw-glass {
    border: 1px solid var(--cw-border);
    border-radius: 28px;
    background:
        linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.018)),
        rgba(7,9,13,.74);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.055),
        0 28px 90px rgba(0,0,0,.28);
    backdrop-filter: blur(18px);
}

.cw-section {
    margin-top: 34px;
}

.cw-section-title {
    font-size: 1.45rem;
    font-weight: 850;
    letter-spacing: -.04em;
    margin-bottom: 5px;
}

.cw-section-subtitle {
    color: var(--cw-muted);
    font-size: .88rem;
    line-height: 1.6;
    margin-bottom: 22px;
}

.cw-metric {
    padding: 20px;
    min-height: 128px;
}

.cw-metric-value {
    font-size: 2rem;
    font-weight: 850;
    letter-spacing: -.05em;
}

.cw-metric-label {
    color: var(--cw-muted);
    font-size: .78rem;
    margin-top: 8px;
}

.cw-form-shell {
    padding: 26px;
}

[data-testid="stNumberInput"] > div,
[data-testid="stSelectbox"] > div {
    border-radius: 14px !important;
}

[data-baseweb="select"] > div {
    background: #171a22 !important;
    border-color: rgba(255,255,255,.08) !important;
    border-radius: 13px !important;
}

[data-testid="stNumberInput"] input {
    background: #171a22 !important;
    color: white !important;
    border-radius: 13px !important;
}

[data-testid="stNumberInput"] button {
    background: transparent !important;
    color: #ffffff !important;
    border: none !important;
}

button[kind="primary"],
button[kind="secondary"] {
    border-radius: 13px !important;
    min-height: 46px !important;
    font-weight: 800 !important;
    letter-spacing: -.01em;
}

button[kind="primary"] {
    background: #f1f3f6 !important;
    color: #08090c !important;
    border: 1px solid #ffffff !important;
}

button[kind="secondary"] {
    background: rgba(255,255,255,.055) !important;
    color: #f4f5f7 !important;
    border: 1px solid rgba(255,255,255,.13) !important;
}

.cw-result {
    margin-top: 30px;
    padding: 42px 34px;
    text-align: center;
    border-radius: 30px;
    border: 1px solid rgba(255,255,255,.11);
    background:
        radial-gradient(circle at 50% 0%, rgba(255,255,255,.09), transparent 45%),
        linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.018));
    position: relative;
    overflow: hidden;
}

.cw-result-approved {
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.07),
        0 25px 100px rgba(78,255,174,.08);
}

.cw-result-rejected {
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.07),
        0 25px 100px rgba(255,80,100,.08);
}

.cw-result-orb {
    width: 120px;
    height: 120px;
    margin: 0 auto 20px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-size: 3.5rem;
    border: 1px solid rgba(255,255,255,.16);
    background:
        radial-gradient(circle at 35% 25%, rgba(255,255,255,.22), rgba(255,255,255,.025));
    box-shadow:
        inset 0 0 45px rgba(255,255,255,.04),
        0 20px 60px rgba(0,0,0,.4);
    animation: cwFloat 3.2s ease-in-out infinite;
}

@keyframes cwFloat {
    0%, 100% { transform: translateY(0) rotateX(0deg); }
    50% { transform: translateY(-8px) rotateX(7deg); }
}

.cw-result-title {
    font-size: clamp(2rem, 4vw, 3.5rem);
    font-weight: 900;
    letter-spacing: -.065em;
}

.cw-result-probability {
    margin-top: 8px;
    color: #b8bec8;
    font-size: .95rem;
}

.cw-footer {
    margin-top: 48px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,.07);
    color: #6f7580;
    text-align: center;
    font-size: .75rem;
}

.stAlert {
    border-radius: 15px !important;
}

</style>
"""


st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ============================================================
# CINEMATIC INTRO
# ============================================================

INTRO_HTML = r"""
<div id="cw-intro-root">
<style>
#cw-intro-root {
    --intro-white: #f7f7f8;
    --intro-soft: #a6abb5;
    --intro-muted: #656b76;
    position: fixed;
    inset: 0;
    z-index: 999999;
    overflow: hidden;
    background:
        radial-gradient(circle at 50% 46%, rgba(255,255,255,.075), transparent 24%),
        radial-gradient(circle at 50% 50%, rgba(95,105,125,.06), transparent 46%),
        #020306;
    color: var(--intro-white);
    display: flex;
    align-items: center;
    justify-content: center;
    perspective: 1400px;
}

#cw-intro-root.cw-intro-exit {
    animation: cwIntroExit 1.15s cubic-bezier(.77,0,.18,1) forwards;
    pointer-events: none;
}

@keyframes cwIntroExit {
    0% {
        opacity: 1;
        transform: scale(1);
        filter: blur(0);
    }
    100% {
        opacity: 0;
        transform: scale(1.08);
        filter: blur(12px);
    }
}

#cw-particle-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    opacity: .7;
}

.cw-space-grid {
    position: absolute;
    width: 150vw;
    height: 150vh;
    left: -25vw;
    top: -25vh;
    opacity: .13;
    background-image:
        linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px);
    background-size: 72px 72px;
    transform:
        perspective(700px)
        rotateX(66deg)
        translateY(34vh)
        translateZ(-100px);
    mask-image: linear-gradient(to bottom, transparent 0%, black 42%, transparent 100%);
}

.cw-camera {
    position: relative;
    z-index: 10;
    width: min(94vw, 1120px);
    min-height: 720px;
    display: flex;
    align-items: center;
    justify-content: center;
    transform-style: preserve-3d;
    transition: transform .18s ease-out;
}

.cw-scene {
    position: relative;
    width: 100%;
    min-height: 720px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transform-style: preserve-3d;
}

.cw-orbit {
    position: absolute;
    left: 50%;
    top: 42%;
    width: min(76vw, 850px);
    height: min(28vw, 310px);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 50%;
    transform:
        translate(-50%,-50%)
        rotateX(67deg)
        rotateZ(-10deg)
        translateZ(-100px);
    box-shadow: 0 0 60px rgba(255,255,255,.025);
    animation: cwOrbitSpin 13s linear infinite;
}

.cw-orbit::before,
.cw-orbit::after {
    content: "";
    position: absolute;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 0 25px rgba(255,255,255,.75);
}

.cw-orbit::before {
    left: 15%;
    top: 10%;
}

.cw-orbit::after {
    right: 12%;
    bottom: 12%;
    width: 6px;
    height: 6px;
    opacity: .6;
}

@keyframes cwOrbitSpin {
    from { transform: translate(-50%,-50%) rotateX(67deg) rotateZ(-10deg) rotateZ(0deg) translateZ(-100px); }
    to { transform: translate(-50%,-50%) rotateX(67deg) rotateZ(-10deg) rotateZ(360deg) translateZ(-100px); }
}

.cw-card-stage {
    position: relative;
    width: min(78vw, 650px);
    height: 390px;
    display: flex;
    align-items: center;
    justify-content: center;
    transform-style: preserve-3d;
    z-index: 20;
}

.cw-card-glow {
    position: absolute;
    width: 72%;
    height: 52%;
    border-radius: 50%;
    background: rgba(255,255,255,.12);
    filter: blur(70px);
    opacity: .32;
    transform: translateZ(-70px);
}

.cw-credit-card {
    position: relative;
    width: min(72vw, 520px);
    aspect-ratio: 1.586 / 1;
    border-radius: 27px;
    transform-style: preserve-3d;
    transform:
        rotateX(9deg)
        rotateY(-9deg)
        rotateZ(-1.5deg);
    animation:
        cwCardReveal 2.8s cubic-bezier(.2,.82,.2,1) forwards,
        cwCardFloat 5s ease-in-out 2.8s infinite;
    box-shadow:
        0 42px 90px rgba(0,0,0,.65),
        0 0 0 1px rgba(255,255,255,.13),
        inset 0 1px 0 rgba(255,255,255,.2);
    overflow: hidden;
}

@keyframes cwCardReveal {
    0% {
        opacity: 0;
        transform:
            translateY(90px)
            translateZ(-260px)
            rotateX(30deg)
            rotateY(-20deg)
            rotateZ(-8deg)
            scale(.72);
    }
    55% {
        opacity: 1;
        transform:
            translateY(-16px)
            translateZ(80px)
            rotateX(8deg)
            rotateY(-8deg)
            rotateZ(-1deg)
            scale(1.02);
    }
    100% {
        opacity: 1;
        transform:
            translateY(0)
            translateZ(0)
            rotateX(9deg)
            rotateY(-9deg)
            rotateZ(-1.5deg)
            scale(1);
    }
}

@keyframes cwCardFloat {
    0%, 100% {
        transform:
            translateY(0)
            rotateX(9deg)
            rotateY(-9deg)
            rotateZ(-1.5deg);
    }
    50% {
        transform:
            translateY(-11px)
            rotateX(11deg)
            rotateY(7deg)
            rotateZ(1deg);
    }
}

.cw-card-face {
    position: absolute;
    inset: 0;
    border-radius: inherit;
    overflow: hidden;
    backface-visibility: hidden;
    background:
        radial-gradient(circle at 18% 15%, rgba(255,255,255,.22), transparent 26%),
        radial-gradient(circle at 82% 80%, rgba(255,255,255,.07), transparent 34%),
        linear-gradient(135deg, #292d35 0%, #15181e 47%, #090b0f 100%);
    border: 1px solid rgba(255,255,255,.17);
}

.cw-card-face::before {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: inherit;
    background:
        linear-gradient(
            115deg,
            transparent 0%,
            rgba(255,255,255,.09) 35%,
            transparent 48%
        );
    background-size: 230% 100%;
    animation: cwCardSheen 4.2s ease-in-out infinite;
}

@keyframes cwCardSheen {
    0%, 100% { background-position: 140% 0; }
    48%, 60% { background-position: -40% 0; }
}

.cw-card-face::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(
            135deg,
            rgba(255,255,255,.018) 0,
            rgba(255,255,255,.018) 1px,
            transparent 1px,
            transparent 8px
        );
    opacity: .4;
}

.cw-card-inner {
    position: absolute;
    inset: 0;
    z-index: 2;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transform: translateZ(30px);
}

.cw-card-logo {
    font-size: clamp(2.2rem, 6vw, 4.6rem);
    line-height: 1;
    font-weight: 950;
    letter-spacing: -.085em;
    text-transform: none;
    color: #f7f8fa;
    text-shadow:
        0 2px 0 rgba(0,0,0,.35),
        0 14px 35px rgba(0,0,0,.55);
}

.cw-card-line {
    width: 95px;
    height: 1px;
    margin-top: 20px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.65), transparent);
}

.cw-card-caption {
    margin-top: 13px;
    color: rgba(255,255,255,.43);
    font-size: .62rem;
    letter-spacing: .32em;
    text-transform: uppercase;
}

.cw-card-depth {
    position: absolute;
    inset: 5px -8px -7px 8px;
    border-radius: 28px;
    z-index: -1;
    background: #080a0e;
    border: 1px solid rgba(255,255,255,.06);
    transform: translateZ(-12px);
    opacity: .85;
}

.cw-credit-card:hover {
    animation-play-state: paused;
    transform:
        rotateX(5deg)
        rotateY(8deg)
        rotateZ(0deg)
        translateY(-8px)
        scale(1.025);
}

.cw-credit-card:hover .cw-card-face {
    box-shadow: inset 0 0 70px rgba(255,255,255,.045);
}

.cw-creator {
    position: relative;
    z-index: 30;
    margin-top: 2px;
    text-align: center;
    opacity: 0;
    animation: cwCreatorIn 1.2s ease 2.15s forwards;
}

@keyframes cwCreatorIn {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.cw-creator-label {
    color: #7c828d;
    font-size: .64rem;
    letter-spacing: .35em;
    text-transform: uppercase;
}

.cw-creator-name {
    margin-top: 8px;
    margin-bottom: 2px;
    font-size: clamp(1rem, 2vw, 1.35rem);
    font-weight: 750;
    letter-spacing: -.025em;
    color: #e9ebee;
}

.cw-quote-zone {
    position: relative;
    width: min(90vw, 850px);
    min-height: 76px;
    margin-top: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    z-index: 40;
}

.cw-quote {
    position: absolute;
    width: 100%;
    color: #c9cdd5;
    font-size: clamp(.82rem, 1.45vw, 1.04rem);
    line-height: 1.55;
    letter-spacing: .02em;
    opacity: 0;
    transform: translateY(10px);
    transition: opacity .65s ease, transform .65s ease;
}

.cw-quote.active {
    opacity: 1;
    transform: translateY(0);
}

.cw-quote strong {
    color: #ffffff;
    font-weight: 800;
}

.cw-progress {
    position: absolute;
    left: 50%;
    bottom: 20px;
    width: min(310px, 55vw);
    height: 1px;
    transform: translateX(-50%);
    background: rgba(255,255,255,.08);
    overflow: hidden;
    z-index: 50;
}

.cw-progress-bar {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, transparent, #ffffff, transparent);
    animation: cwProgress 11.5s linear forwards;
}

@keyframes cwProgress {
    from { width: 0%; }
    to { width: 100%; }
}

.cw-enter-hint {
    position: absolute;
    right: 28px;
    bottom: 24px;
    z-index: 60;
    color: #656b75;
    font-size: .62rem;
    letter-spacing: .2em;
    text-transform: uppercase;
}

.cw-3d-sphere {
    position: absolute;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background:
        radial-gradient(circle at 32% 26%, #ffffff 0, #737984 7%, #20242c 34%, #07080b 72%);
    box-shadow:
        0 0 35px rgba(255,255,255,.12),
        inset -8px -8px 15px rgba(0,0,0,.65);
    transform: translateZ(170px);
    animation: cwSphere 7s ease-in-out infinite;
}

@keyframes cwSphere {
    0%, 100% {
        left: 12%;
        top: 24%;
    }
    50% {
        left: 82%;
        top: 29%;
    }
}

.cw-ring {
    position: absolute;
    width: 100px;
    height: 100px;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 50%;
    transform-style: preserve-3d;
    animation: cwRing 9s linear infinite;
}

.cw-ring.r1 {
    left: 9%;
    bottom: 28%;
    transform: rotateX(70deg) rotateY(12deg);
}

.cw-ring.r2 {
    right: 10%;
    top: 25%;
    width: 75px;
    height: 75px;
    transform: rotateY(70deg) rotateZ(22deg);
    animation-duration: 12s;
}

@keyframes cwRing {
    to { transform: rotateX(70deg) rotateY(12deg) rotateZ(360deg); }
}

@media (max-width: 700px) {
    .cw-card-stage {
        height: 300px;
    }

    .cw-scene {
        min-height: 640px;
    }

    .cw-orbit {
        width: 120vw;
    }

    .cw-creator {
        margin-top: 0;
    }

    .cw-enter-hint {
        display: none;
    }
}

@media (prefers-reduced-motion: reduce) {
    #cw-intro-root *,
    #cw-intro-root *::before,
    #cw-intro-root *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}
</style>

<canvas id="cw-particle-canvas" aria-hidden="true"></canvas>

<div class="cw-space-grid"></div>

<div class="cw-camera" id="cw-camera">
    <div class="cw-scene">

        <div class="cw-orbit"></div>

        <div class="cw-ring r1"></div>
        <div class="cw-ring r2"></div>
        <div class="cw-3d-sphere"></div>

        <div class="cw-card-stage">
            <div class="cw-card-glow"></div>

            <div class="cw-credit-card" id="cw-credit-card">
                <div class="cw-card-depth"></div>

                <div class="cw-card-face">
                    <div class="cw-card-inner">
                        <div class="cw-card-logo">CreditWise</div>
                        <div class="cw-card-line"></div>
                        <div class="cw-card-caption">
                            Intelligent lending decisions
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="cw-creator">
            <div class="cw-creator-label">
                Architected &amp; Engineered by
            </div>
            <div class="cw-creator-name">
                Gurram Indrasena Yadav
            </div>
        </div>

        <div class="cw-quote-zone" aria-live="polite">
            <div class="cw-quote active">
                <strong>Analyze everything.</strong>
                Decide with confidence.
            </div>

            <div class="cw-quote">
                Read the <strong>risk</strong>. Understand the borrower.
            </div>

            <div class="cw-quote">
                Turn financial data into <strong>clear decisions</strong>.
            </div>

            <div class="cw-quote">
                Measure the <strong>scale</strong>. Understand the signal.
            </div>

            <div class="cw-quote">
                Better lending begins with <strong>better insight</strong>.
            </div>
        </div>

        <div class="cw-progress">
            <div class="cw-progress-bar"></div>
        </div>

        <div class="cw-enter-hint">
            Credit intelligence · Loan prediction · Risk analysis
        </div>

    </div>
</div>

<script>
(function () {
    const root = document.getElementById("cw-intro-root");
    const camera = document.getElementById("cw-camera");
    const canvas = document.getElementById("cw-particle-canvas");

    if (!root || !camera || !canvas) return;

    let closed = false;

    function closeIntro() {
        if (closed) return;
        closed = true;

        root.classList.add("cw-intro-exit");

        window.setTimeout(function () {
            root.style.display = "none";
        }, 1200);
    }

    window.setTimeout(closeIntro, 12000);

    // --------------------------------------------------------
    // Mouse-driven 3D camera movement.
    // It is intentionally subtle so the card never turns
    // completely side-on.
    // --------------------------------------------------------
    let targetX = 0;
    let targetY = 0;
    let currentX = 0;
    let currentY = 0;

    root.addEventListener("pointermove", function (event) {
        const rect = root.getBoundingClientRect();

        const px = (event.clientX - rect.left) / rect.width;
        const py = (event.clientY - rect.top) / rect.height;

        targetX = (px - 0.5) * 5;
        targetY = (py - 0.5) * -3;
    });

    function animateCamera() {
        currentX += (targetX - currentX) * 0.035;
        currentY += (targetY - currentY) * 0.035;

        camera.style.transform =
            "rotateY(" + currentX + "deg) " +
            "rotateX(" + currentY + "deg)";

        window.requestAnimationFrame(animateCamera);
    }

    animateCamera();

    // --------------------------------------------------------
    // Quote rotation.
    // --------------------------------------------------------
    const quotes = Array.from(root.querySelectorAll(".cw-quote"));
    let quoteIndex = 0;

    window.setInterval(function () {
        if (closed || quotes.length === 0) return;

        quotes[quoteIndex].classList.remove("active");
        quoteIndex = (quoteIndex + 1) % quotes.length;
        quotes[quoteIndex].classList.add("active");
    }, 1800);

    // --------------------------------------------------------
    // 3D particle field.
    // --------------------------------------------------------
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = 0;
    let height = 0;
    let animationFrame = 0;

    const particles = [];
    const particleCount = Math.min(
        150,
        Math.max(70, Math.floor(window.innerWidth / 9))
    );

    function resizeCanvas() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    function makeParticle() {
        return {
            x: (Math.random() - 0.5) * width * 1.7,
            y: (Math.random() - 0.5) * height * 1.7,
            z: Math.random() * 1 + 0.1,
            speed: 0.001 + Math.random() * 0.003,
            size: 0.45 + Math.random() * 1.5
        };
    }

    resizeCanvas();

    for (let i = 0; i < particleCount; i += 1) {
        particles.push(makeParticle());
    }

    function drawParticles() {
        if (closed) return;

        ctx.clearRect(0, 0, width, height);

        const cx = width / 2;
        const cy = height / 2;

        for (let i = 0; i < particles.length; i += 1) {
            const p = particles[i];

            p.z -= p.speed;

            if (p.z <= 0.02) {
                p.x = (Math.random() - 0.5) * width * 1.7;
                p.y = (Math.random() - 0.5) * height * 1.7;
                p.z = 1;
            }

            const perspective = 1 / p.z;
            const sx = cx + p.x * perspective * 0.36;
            const sy = cy + p.y * perspective * 0.36;

            if (
                sx < -50 ||
                sx > width + 50 ||
                sy < -50 ||
                sy > height + 50
            ) {
                continue;
            }

            const radius = Math.max(0.3, p.size * perspective * 0.45);
            const alpha = Math.min(0.65, 0.08 + (1 - p.z) * 0.55);

            ctx.beginPath();
            ctx.fillStyle = "rgba(255,255,255," + alpha + ")";
            ctx.arc(sx, sy, radius, 0, Math.PI * 2);
            ctx.fill();
        }

        animationFrame = window.requestAnimationFrame(drawParticles);
    }

    window.addEventListener("resize", resizeCanvas);
    drawParticles();

    window.setTimeout(function () {
        if (animationFrame) {
            window.cancelAnimationFrame(animationFrame);
        }
    }, 12500);
})();
</script>
"""


if not st.session_state["intro_seen"]:
    st.html(INTRO_HTML, unsafe_allow_javascript=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### CreditWise")
    st.caption("Intelligent loan approval prediction")

    st.divider()

    st.markdown("**Deployment model**")
    st.caption("Logistic Regression")

    st.markdown("**Input features**")
    st.caption(f"{len(MODEL_INPUT_COLUMNS)} raw features")

    st.markdown("**Preprocessing**")
    st.caption("Imputation + scaling + one-hot encoding")

    st.markdown("**Decision**")
    st.caption("Loan approval probability")


# ============================================================
# MAIN BRAND / HERO
# ============================================================

st.markdown(
    """
    <div class="cw-topbar">
        <div class="cw-brand">
            <div class="cw-brand-mark">💳</div>
            <div>
                <div class="cw-brand-name">CreditWise</div>
                <div class="cw-brand-sub">
                    Intelligent loan decision engine
                </div>
            </div>
        </div>
        <div class="cw-kicker">
            Financial intelligence · ML prediction
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cw-kicker">Predict the decision before the decision</div>
    <div class="cw-hero-title">Know your loan.</div>
    <div class="cw-hero-title">Understand the risk.</div>
    <div class="cw-hero-copy">
        CreditWise evaluates borrower information through a trained machine
        learning pipeline and estimates the probability of loan approval.
        Enter the applicant profile below and let the model analyze the
        financial signal.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.markdown('<div class="cw-section">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="cw-section-title">Model intelligence</div>
    <div class="cw-section-subtitle">
        The deployed model receives the same 18 raw features used during
        training. Preprocessing is already packaged inside model.pkl.
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)

metrics = [
    ("18", "Raw input features"),
    ("88.95%", "Validation accuracy"),
    ("81.74%", "Validation F1"),
    ("78.33%", "Approval recall"),
]

for column, (value, label) in zip(metric_cols, metrics):
    with column:
        st.markdown(
            f"""
            <div class="cw-glass cw-metric">
                <div class="cw-metric-value">{value}</div>
                <div class="cw-metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# INPUT FORM
# ============================================================

st.markdown('<div class="cw-section">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="cw-section-title">Applicant profile</div>
    <div class="cw-section-subtitle">
        Provide the applicant's financial, employment and loan information.
        No extra engineered columns are created in the frontend.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("creditwise_prediction_form", clear_on_submit=False):

    st.markdown(
        '<div class="cw-glass cw-form-shell">',
        unsafe_allow_html=True,
    )

    st.markdown("#### Financial profile")

    row1 = st.columns(3)

    with row1[0]:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f",
        )

    with row1[1]:
        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            format="%.2f",
        )

    with row1[2]:
        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f",
        )

    row2 = st.columns(3)

    with row2[0]:
        collateral_value = st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=100000.0,
            step=5000.0,
            format="%.2f",
        )

    with row2[1]:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=100000.0,
            step=5000.0,
            format="%.2f",
        )

    with row2[2]:
        loan_term = st.number_input(
            "Loan Term",
            min_value=1.0,
            value=12.0,
            step=1.0,
            format="%.0f",
        )

    st.markdown("#### Credit & borrower profile")

    row3 = st.columns(4)

    with row3[0]:
        age = st.number_input(
            "Age",
            min_value=18.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            format="%.0f",
        )

    with row3[1]:
        dependents = st.number_input(
            "Dependents",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            format="%.0f",
        )

    with row3[2]:
        credit_score = st.number_input(
            "Credit Score",
            min_value=0.0,
            max_value=1000.0,
            value=700.0,
            step=10.0,
            format="%.0f",
        )

    with row3[3]:
        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=1.0,
            format="%.0f",
        )

    row4 = st.columns(3)

    with row4[0]:
        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            format="%.2f",
        )

    with row4[1]:
        employment_status = st.selectbox(
            "Employment Status",
            CATEGORY_OPTIONS["Employment_Status"],
        )

    with row4[2]:
        marital_status = st.selectbox(
            "Marital Status",
            CATEGORY_OPTIONS["Marital_Status"],
        )

    st.markdown("#### Loan details")

    row5 = st.columns(4)

    with row5[0]:
        loan_purpose = st.selectbox(
            "Loan Purpose",
            CATEGORY_OPTIONS["Loan_Purpose"],
        )

    with row5[1]:
        property_area = st.selectbox(
            "Property Area",
            CATEGORY_OPTIONS["Property_Area"],
        )

    with row5[2]:
        education_level = st.selectbox(
            "Education Level",
            CATEGORY_OPTIONS["Education_Level"],
        )

    with row5[3]:
        gender = st.selectbox(
            "Gender",
            CATEGORY_OPTIONS["Gender"],
        )

    row6 = st.columns(3)

    with row6[0]:
        employer_category = st.selectbox(
            "Employer Category",
            CATEGORY_OPTIONS["Employer_Category"],
        )

    with row6[1]:
        st.empty()

    with row6[2]:
        st.empty()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    submit = st.form_submit_button(
        "CHECK LOAN ELIGIBILITY",
        type="primary",
        use_container_width=False,
    )

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION
# ============================================================

if submit:

    raw_values = {
        "Applicant_Income": float(applicant_income),
        "Coapplicant_Income": float(coapplicant_income),
        "Employment_Status": employment_status,
        "Age": float(age),
        "Marital_Status": marital_status,
        "Dependents": float(dependents),
        "Credit_Score": float(credit_score),
        "Existing_Loans": float(existing_loans),
        "DTI_Ratio": float(dti_ratio),
        "Savings": float(savings),
        "Collateral_Value": float(collateral_value),
        "Loan_Amount": float(loan_amount),
        "Loan_Term": float(loan_term),
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }

    # --------------------------------------------------------
    # 3D PROCESSING SCENE
    # --------------------------------------------------------

    PROCESSING_HTML = r"""
    <style>
    .cw-processing {
        position: relative;
        min-height: 410px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.11);
        border-radius: 30px;
        background:
            radial-gradient(circle at 50% 42%, rgba(255,255,255,.08), transparent 24%),
            linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.012));
        perspective: 900px;
    }

    .cw-processing-scene {
        position: relative;
        width: 240px;
        height: 240px;
        transform-style: preserve-3d;
        animation: cwProcessCamera 4s ease-in-out infinite;
    }

    @keyframes cwProcessCamera {
        0%,100% { transform: rotateX(9deg) rotateY(-12deg); }
        50% { transform: rotateX(-5deg) rotateY(12deg); }
    }

    .cw-process-cube {
        position: absolute;
        inset: 48px;
        transform-style: preserve-3d;
        animation: cwCube 3.2s linear infinite;
    }

    @keyframes cwCube {
        to { transform: rotateX(360deg) rotateY(360deg); }
    }

    .cw-process-face {
        position: absolute;
        inset: 0;
        border: 1px solid rgba(255,255,255,.25);
        background: rgba(255,255,255,.025);
        box-shadow: inset 0 0 35px rgba(255,255,255,.035);
    }

    .cw-process-face.front  { transform: translateZ(72px); }
    .cw-process-face.back   { transform: rotateY(180deg) translateZ(72px); }
    .cw-process-face.right  { transform: rotateY(90deg) translateZ(72px); }
    .cw-process-face.left   { transform: rotateY(-90deg) translateZ(72px); }
    .cw-process-face.top    { transform: rotateX(90deg) translateZ(72px); }
    .cw-process-face.bottom { transform: rotateX(-90deg) translateZ(72px); }

    .cw-process-ring {
        position: absolute;
        inset: 18px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.18);
        transform: rotateX(70deg);
        animation: cwRingSpin 2.3s linear infinite;
    }

    .cw-process-ring.r2 {
        inset: 35px;
        transform: rotateY(70deg);
        animation-duration: 1.7s;
    }

    @keyframes cwRingSpin {
        to { transform: rotateX(70deg) rotateZ(360deg); }
    }

    .cw-processing-copy {
        position: absolute;
        left: 0;
        right: 0;
        bottom: 40px;
        text-align: center;
    }

    .cw-processing-title {
        font-size: 1.35rem;
        font-weight: 850;
        letter-spacing: -.04em;
    }

    .cw-processing-subtitle {
        margin-top: 8px;
        color: #858b96;
        font-size: .78rem;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .cw-processing-dots span {
        display: inline-block;
        width: 5px;
        height: 5px;
        margin: 0 3px;
        border-radius: 50%;
        background: #ffffff;
        opacity: .2;
        animation: cwDot 1.2s infinite;
    }

    .cw-processing-dots span:nth-child(2) { animation-delay: .18s; }
    .cw-processing-dots span:nth-child(3) { animation-delay: .36s; }

    @keyframes cwDot {
        0%,100% { opacity: .2; transform: translateY(0); }
        50% { opacity: 1; transform: translateY(-5px); }
    }
    </style>

    <div class="cw-processing" id="cw-processing-scene">
        <div class="cw-processing-scene">
            <div class="cw-process-ring"></div>
            <div class="cw-process-ring r2"></div>

            <div class="cw-process-cube">
                <div class="cw-process-face front"></div>
                <div class="cw-process-face back"></div>
                <div class="cw-process-face right"></div>
                <div class="cw-process-face left"></div>
                <div class="cw-process-face top"></div>
                <div class="cw-process-face bottom"></div>
            </div>
        </div>

        <div class="cw-processing-copy">
            <div class="cw-processing-title">
                Analyzing the financial signal
            </div>
            <div class="cw-processing-subtitle">
                Credit profile · Risk · Loan decision
            </div>
            <div class="cw-processing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    </div>

    <script>
    (function () {
        const scene = document.getElementById("cw-processing-scene");
        if (!scene) return;

        window.setTimeout(function () {
            scene.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });
        }, 80);
    })();
    </script>
    """

    st.html(PROCESSING_HTML, unsafe_allow_javascript=True)

    # Give the 3D checking scene enough time to be visible.
    # The scene is rendered below the form and the browser is automatically
    # scrolled down to it by the JavaScript above.
    time.sleep(2.8)

    try:
        prediction, probability, input_df = predict_loan(raw_values)

        # Store the real result in Streamlit session state.
        st.session_state["prediction"] = prediction
        st.session_state["probability"] = probability
        st.session_state["last_input"] = raw_values

        # Rerun so the processing scene is replaced by the result scene.
        st.rerun()

    except Exception as exc:
        st.error("CreditWise could not complete the prediction.")
        st.code(str(exc))

        # Helpful diagnostic information without exposing internals
        # in the normal UI.
        try:
            expected = get_model_expected_columns(model)
            st.caption("Model expects these columns:")
            st.code("\n".join(expected))
        except Exception:
            pass


# ============================================================
# RESULT SCENE
# ============================================================

prediction = st.session_state.get("prediction")
probability = st.session_state.get("probability")


if prediction is not None:

    if prediction == 1:
        result_title = "Loan Approved"
        result_emoji = "✓"
        result_class = "cw-result-approved"
        result_subtitle = (
            "The model estimates that this application has a "
            "positive approval signal."
        )
        result_color = "rgba(124,247,189,.85)"
    else:
        result_title = "Loan Not Approved"
        result_emoji = "×"
        result_class = "cw-result-rejected"
        result_subtitle = (
            "The model estimates that this application has a "
            "negative approval signal."
        )
        result_color = "rgba(255,123,134,.85)"

    probability_text = (
        f"{probability * 100:.2f}% estimated approval probability"
        if probability is not None
        else "Probability unavailable"
    )

    RESULT_HTML = f"""
    <style>
    .cw-result-scene {{
        position: relative;
        min-height: 430px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border-radius: 32px;
        border: 1px solid rgba(255,255,255,.12);
        background:
            radial-gradient(circle at 50% 35%, rgba(255,255,255,.085), transparent 25%),
            linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.012));
        perspective: 1000px;
    }}

    .cw-result-core {{
        position: relative;
        z-index: 3;
        text-align: center;
        transform-style: preserve-3d;
        animation: cwResultEnter 1.1s cubic-bezier(.2,.8,.2,1) forwards;
    }}

    @keyframes cwResultEnter {{
        from {{
            opacity: 0;
            transform:
                translateY(40px)
                rotateX(24deg)
                scale(.82);
        }}
        to {{
            opacity: 1;
            transform:
                translateY(0)
                rotateX(0)
                scale(1);
        }}
    }}

    .cw-result-symbol {{
        width: 128px;
        height: 128px;
        margin: 0 auto 22px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        font-size: 4.3rem;
        font-weight: 300;
        color: #ffffff;
        border: 1px solid rgba(255,255,255,.17);
        background:
            radial-gradient(
                circle at 35% 25%,
                rgba(255,255,255,.22),
                rgba(255,255,255,.018) 58%
            );
        box-shadow:
            inset 0 0 55px rgba(255,255,255,.045),
            0 22px 75px rgba(0,0,0,.48),
            0 0 55px {result_color};
        animation: cwResultFloat 3s ease-in-out infinite;
    }}

    @keyframes cwResultFloat {{
        0%,100% {{
            transform:
                translateY(0)
                rotateX(0)
                rotateY(0);
        }}
        50% {{
            transform:
                translateY(-10px)
                rotateX(7deg)
                rotateY(8deg);
        }}
    }}

    .cw-result-heading {{
        font-size: clamp(2.1rem, 5vw, 4.2rem);
        font-weight: 950;
        letter-spacing: -.075em;
    }}

    .cw-result-copy {{
        max-width: 620px;
        margin: 10px auto 0;
        color: #9ca2ad;
        line-height: 1.65;
        font-size: .9rem;
    }}

    .cw-result-prob {{
        margin-top: 17px;
        font-size: 1.02rem;
        font-weight: 750;
        color: #e6e8ec;
    }}

    .cw-result-particle {{
        position: absolute;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #ffffff;
        opacity: .28;
        animation: cwParticleFloat 4s ease-in-out infinite;
    }}

    .cw-result-particle.p1 {{
        left: 14%;
        top: 28%;
    }}

    .cw-result-particle.p2 {{
        left: 26%;
        bottom: 23%;
        animation-delay: .6s;
    }}

    .cw-result-particle.p3 {{
        right: 17%;
        top: 24%;
        animation-delay: 1.2s;
    }}

    .cw-result-particle.p4 {{
        right: 28%;
        bottom: 18%;
        animation-delay: 1.7s;
    }}

    @keyframes cwParticleFloat {{
        0%,100% {{
            transform: translate3d(0,0,0) scale(1);
        }}
        50% {{
            transform: translate3d(0,-24px,40px) scale(1.6);
        }}
    }}
    </style>

    <div class="cw-result-scene" id="cw-result-scene">
        <div class="cw-result-particle p1"></div>
        <div class="cw-result-particle p2"></div>
        <div class="cw-result-particle p3"></div>
        <div class="cw-result-particle p4"></div>

        <div class="cw-result-core">
            <div class="cw-result-symbol">{result_emoji}</div>
            <div class="cw-result-heading">{result_title}</div>
            <div class="cw-result-copy">
                {result_subtitle}
            </div>
            <div class="cw-result-prob">
                {probability_text}
            </div>
        </div>
    </div>

    <script>
    (function () {{
        const scene = document.getElementById("cw-result-scene");
        if (!scene) return;

        window.setTimeout(function () {{
            scene.scrollIntoView({{
                behavior: "smooth",
                block: "center"
            }});
        }}, 100);
    }})();
    </script>
    """

    st.html(RESULT_HTML, unsafe_allow_javascript=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    result_cols = st.columns([1, 1, 1])

    with result_cols[1]:
        if st.button(
            "RUN ANOTHER APPLICATION",
            type="secondary",
            use_container_width=True,
        ):
            reset_result()
            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="cw-footer">
        CreditWise · Machine-learning based loan approval prediction
        · Engineered by Gurram Indrasena Yadav
    </div>
    """,
    unsafe_allow_html=True,
)
