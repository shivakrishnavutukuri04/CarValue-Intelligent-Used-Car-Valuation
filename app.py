
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CarValue — Intelligent Vehicle Valuation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #11130f;
        --paper: #f4f1e9;
        --card: #fffdf7;
        --line: #d8d3c7;
        --muted: #74756d;
        --acid: #c8ff39;
        --acid-dark: #8ebc12;
        --navy: #17202b;
        --white: #fffdf7;
    }

    html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 12%, rgba(200,255,57,.13), transparent 23%),
            linear-gradient(135deg, #f4f1e9 0%, #eeece4 58%, #e5e3db 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244,241,233,.82);
    }

    .block-container {
        max-width: 1400px;
        padding: 2rem 3.5rem 4rem 3.5rem;
    }

    /* Remove default Streamlit clutter */
    #MainMenu, footer { visibility: hidden; }

    /* Top navigation */
    .topbar {
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding: 5px 0 24px 0;
        border-bottom: 1px solid rgba(17,19,15,.13);
        margin-bottom: 36px;
    }

    .brand {
        display:flex;
        align-items:center;
        gap:11px;
        font-family:"Manrope",sans-serif;
        font-weight:800;
        font-size:20px;
        letter-spacing:-.5px;
    }

    .brand-mark {
        width:32px;
        height:32px;
        border-radius:9px;
        background:var(--ink);
        color:var(--acid);
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:17px;
    }

    .nav-meta {
        display:flex;
        align-items:center;
        gap:18px;
        color:#62635c;
        font-size:12px;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:1.2px;
    }

    .status-dot {
        width:7px;
        height:7px;
        border-radius:50%;
        background:#7cab00;
        display:inline-block;
        box-shadow:0 0 0 4px rgba(124,171,0,.12);
    }

    /* Hero */
    .hero-grid {
        display:grid;
        grid-template-columns: 1.22fr .78fr;
        gap: 36px;
        align-items:end;
        margin-bottom: 34px;
    }

    .eyebrow {
        color:#62635c;
        text-transform:uppercase;
        letter-spacing:2.4px;
        font-size:11px;
        font-weight:700;
        margin-bottom:13px;
    }

    .hero-title {
        font-family:"Manrope",sans-serif;
        font-weight:800;
        font-size:clamp(42px, 6vw, 78px);
        letter-spacing:-4px;
        line-height:.93;
        margin:0;
        color:var(--ink);
    }

    .hero-title span {
        color:#7da800;
    }

    .hero-copy {
        max-width:650px;
        color:#65665f;
        font-size:16px;
        line-height:1.65;
        margin-top:18px;
    }

    .hero-side {
        background:var(--ink);
        color:var(--white);
        border-radius:24px;
        padding:26px 28px;
        position:relative;
        overflow:hidden;
        min-height:145px;
    }

    .hero-side:after {
        content:"";
        position:absolute;
        width:160px;
        height:160px;
        border:1px solid rgba(200,255,57,.26);
        border-radius:50%;
        right:-35px;
        top:-62px;
    }

    .hero-side-label {
        color:#9ba093;
        text-transform:uppercase;
        font-size:10px;
        letter-spacing:1.8px;
        font-weight:700;
    }

    .hero-side-value {
        font-family:"Manrope",sans-serif;
        font-size:30px;
        font-weight:800;
        margin-top:7px;
    }

    .hero-side-note {
        color:#aeb3a7;
        font-size:12px;
        margin-top:6px;
        max-width:240px;
    }

    /* Section navigation */
    .section-nav {
        display:flex;
        gap:9px;
        margin: 0 0 20px 0;
        overflow-x:auto;
    }

    .nav-pill {
        padding:9px 14px;
        border:1px solid var(--line);
        border-radius:999px;
        font-size:11px;
        font-weight:700;
        color:#66675f;
        background:rgba(255,255,255,.38);
        white-space:nowrap;
    }

    .nav-pill.active {
        background:var(--ink);
        border-color:var(--ink);
        color:var(--acid);
    }

    /* Main workspace */
    .workspace {
        display:grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(310px, .55fr);
        gap:20px;
        align-items:start;
    }

    .panel {
        background:rgba(255,253,247,.88);
        border:1px solid rgba(17,19,15,.12);
        border-radius:24px;
        box-shadow:0 16px 45px rgba(17,19,15,.06);
    }

    .form-panel {
        padding:30px;
    }

    .panel-heading {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:20px;
        padding-bottom:22px;
        border-bottom:1px solid var(--line);
        margin-bottom:25px;
    }

    .panel-title {
        font-family:"Manrope",sans-serif;
        font-size:24px;
        font-weight:800;
        letter-spacing:-1px;
    }

    .panel-subtitle {
        color:var(--muted);
        font-size:13px;
        margin-top:5px;
    }

    .step {
        font-family:"Manrope",sans-serif;
        font-weight:800;
        color:#8a8b82;
        font-size:12px;
        border:1px solid var(--line);
        border-radius:999px;
        padding:7px 11px;
    }

    .field-group {
        margin-bottom:20px;
    }

    .field-label {
        color:#30322c;
        font-size:12px;
        font-weight:700;
        margin-bottom:8px;
        text-transform:uppercase;
        letter-spacing:1px;
    }

    /* Make Streamlit widgets fit the design */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        display:none;
    }

    div[data-baseweb="select"] > div {
        background:#f5f3ec !important;
        border:1px solid #d5d1c6 !important;
        border-radius:12px !important;
        min-height:47px !important;
        color:#171914 !important;
    }

    div[data-testid="stNumberInput"] input {
        background:#f5f3ec !important;
        border:1px solid #d5d1c6 !important;
        border-radius:12px !important;
        min-height:47px !important;
        color:#171914 !important;
    }

    div[data-testid="stNumberInput"] > div {
        background:transparent !important;
    }

    /* Result panel */
    .result-panel {
        background:var(--ink);
        color:var(--white);
        border-radius:24px;
        padding:28px;
        position:sticky;
        top:25px;
        min-height:500px;
        overflow:hidden;
    }

    .result-panel:before {
        content:"";
        position:absolute;
        width:280px;
        height:280px;
        border:1px solid rgba(200,255,57,.13);
        border-radius:50%;
        right:-130px;
        bottom:-100px;
    }

    .result-kicker {
        color:#aeb3a7;
        text-transform:uppercase;
        letter-spacing:2px;
        font-size:10px;
        font-weight:700;
    }

    .car-graphic {
        height:145px;
        display:flex;
        align-items:center;
        justify-content:center;
        margin:4px 0 7px 0;
    }

    .car-outline {
        width:230px;
        height:84px;
        border:3px solid var(--acid);
        border-radius:72px 82px 22px 22px;
        position:relative;
        transform:skewX(-7deg);
        opacity:.92;
    }

    .car-outline:before {
        content:"";
        position:absolute;
        width:108px;
        height:48px;
        border:2px solid var(--acid);
        border-bottom:0;
        border-radius:52px 58px 0 0;
        left:57px;
        top:-48px;
    }

    .car-outline:after {
        content:"●       ●";
        white-space:pre;
        position:absolute;
        color:var(--acid);
        letter-spacing:72px;
        font-size:27px;
        left:39px;
        bottom:-29px;
    }

    .result-caption {
        color:#8f9689;
        font-size:11px;
        text-transform:uppercase;
        letter-spacing:1.6px;
    }

    .result-price {
        font-family:"Manrope",sans-serif;
        color:var(--acid);
        font-size:43px;
        line-height:1;
        font-weight:800;
        letter-spacing:-2px;
        margin:8px 0;
    }

    .result-usd {
        color:#aeb3a7;
        font-size:12px;
    }

    .meter {
        margin:25px 0 10px;
    }

    .meter-head {
        display:flex;
        justify-content:space-between;
        color:#aeb3a7;
        font-size:11px;
        margin-bottom:7px;
    }

    .meter-track {
        height:8px;
        border-radius:99px;
        background:#2b3129;
        overflow:hidden;
    }

    .meter-fill {
        height:100%;
        width:68%;
        background:var(--acid);
        border-radius:99px;
    }

    .spec-grid {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:8px;
        margin-top:23px;
    }

    .spec {
        border:1px solid rgba(255,255,255,.09);
        background:rgba(255,255,255,.035);
        border-radius:12px;
        padding:11px;
    }

    .spec-label {
        color:#81887d;
        font-size:9px;
        text-transform:uppercase;
        letter-spacing:1.2px;
    }

    .spec-value {
        color:#f5f6f0;
        font-size:13px;
        font-weight:700;
        margin-top:3px;
    }

    .result-empty {
        color:#b1b6ab;
        font-size:13px;
        line-height:1.6;
        margin-top:15px;
    }

    /* Primary button */
    div.stButton > button {
        background:var(--acid) !important;
        color:var(--ink) !important;
        border:0 !important;
        border-radius:12px !important;
        min-height:52px !important;
        font-family:"Manrope",sans-serif !important;
        font-size:14px !important;
        font-weight:800 !important;
        letter-spacing:.1px !important;
        box-shadow:0 7px 20px rgba(115,147,0,.16);
    }

    div.stButton > button:hover {
        background:#d5ff69 !important;
        transform:translateY(-1px);
    }

    /* Info pages */
    .info-card {
        background:rgba(255,253,247,.88);
        border:1px solid rgba(17,19,15,.12);
        border-radius:22px;
        padding:26px;
        margin-bottom:16px;
    }

    .info-card h3 {
        font-family:"Manrope",sans-serif;
        margin:0 0 8px;
        color:#171914;
    }

    .info-card p {
        color:#66675f;
        line-height:1.7;
        font-size:14px;
    }

    @media (max-width: 900px) {
        .block-container { padding:1.2rem 1rem 3rem 1rem; }
        .hero-grid, .workspace { grid-template-columns:1fr; }
        .result-panel { position:relative; top:auto; }
        .hero-title { letter-spacing:-2.5px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA / MODEL
# ============================================================
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
USD_TO_INR = 86.0

EXPECTED_COLUMNS = [
    "make_year",
    "mileage_kmpl",
    "engine_cc",
    "fuel_type",
    "owner_count",
    "brand",
    "transmission",
    "color",
    "service_history",
    "accidents_reported",
    "insurance_valid",
]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl is missing from the application folder.")
    return joblib.load(MODEL_PATH)

def esc(value):
    return html.escape(str(value))

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark">◈</div>
            <div>CARVALUE</div>
        </div>
        <div class="nav-meta">
            <span><i class="status-dot"></i>&nbsp; Model Online</span>
            <span>Valuation Engine / 01</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation via radio, visually styled as pills.
page = st.radio(
    "Page",
    ["Valuation", "Methodology", "Model Report"],
    horizontal=True,
    label_visibility="collapsed",
)

# ============================================================
# VALUATION PAGE
# ============================================================
if page == "Valuation":

    st.markdown(
        """
        <div class="hero-grid">
            <div>
                <div class="eyebrow">Used vehicle intelligence / 2026</div>
                <h1 class="hero-title">Know what<br>your car is <span>worth.</span></h1>
                <div class="hero-copy">
                    A machine-learning valuation workspace built from your vehicle's
                    specifications, condition, ownership history and documentation.
                    Enter the details once. Get a clear market estimate.
                </div>
            </div>
            <div class="hero-side">
                <div class="hero-side-label">Prediction engine</div>
                <div class="hero-side-value">11 vehicle signals</div>
                <div class="hero-side-note">Preprocessing and regression are handled by the trained pipeline.</div>
            </div>
        </div>

        <div class="section-nav">
            <div class="nav-pill active">01&nbsp; Vehicle profile</div>
            <div class="nav-pill">02&nbsp; Condition</div>
            <div class="nav-pill">03&nbsp; Documentation</div>
            <div class="nav-pill">04&nbsp; Valuation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Session state for last prediction.
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "prediction_input" not in st.session_state:
        st.session_state.prediction_input = None

    left, right = st.columns([1.45, .55], gap="large")

    with left:
        st.markdown('<div class="panel form-panel">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="panel-heading">
                <div>
                    <div class="panel-title">Vehicle profile</div>
                    <div class="panel-subtitle">Tell us about the car you want to value.</div>
                </div>
                <div class="step">STEP 01 / 04</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="field-label">Identity</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            brand = st.selectbox(
                "Brand", ["Toyota", "Honda", "BMW", "Ford", "Hyundai",
                          "Nissan", "Chevrolet", "Kia", "Volkswagen", "Tesla"],
                key="brand"
            )
        with b:
            make_year = st.number_input(
                "Manufacturing Year", min_value=1995, max_value=2023,
                value=2015, step=1, key="year"
            )
        with c:
            color = st.selectbox(
                "Color", ["White", "Black", "Blue", "Red", "Gray", "Silver"],
                key="color"
            )

        st.markdown('<div class="field-label" style="margin-top:18px;">Powertrain</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"], key="fuel")
        with b:
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"], key="trans")
        with c:
            engine_cc = st.number_input(
                "Engine Capacity (cc)", min_value=500, max_value=5000,
                value=1500, step=50, key="engine"
            )

        st.markdown('<div class="field-label" style="margin-top:18px;">Usage & condition</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            mileage_kmpl = st.number_input(
                "Mileage (kmpl)", min_value=5.0, max_value=35.0,
                value=18.0, step=0.1, key="mileage"
            )
        with b:
            owner_count = st.number_input(
                "Previous Owners", min_value=1, max_value=10,
                value=1, step=1, key="owners"
            )
        with c:
            accidents_reported = st.number_input(
                "Accidents Reported", min_value=0, max_value=10,
                value=0, step=1, key="accidents"
            )

        st.markdown('<div class="field-label" style="margin-top:18px;">Documentation</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            service_history = st.selectbox(
                "Service History", ["Full", "Partial", "Unknown"], key="service"
            )
        with b:
            insurance_valid = st.selectbox(
                "Insurance Valid", ["Yes", "No"], key="insurance"
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("Calculate vehicle value  →", use_container_width=True):
            input_data = pd.DataFrame({
                "make_year": [make_year],
                "mileage_kmpl": [mileage_kmpl],
                "engine_cc": [engine_cc],
                "fuel_type": [fuel_type],
                "owner_count": [owner_count],
                "brand": [brand],
                "transmission": [transmission],
                "color": [color],
                "service_history": [service_history],
                "accidents_reported": [accidents_reported],
                "insurance_valid": [insurance_valid],
            })

            try:
                model = load_model()
                prediction_usd = float(model.predict(input_data)[0])
                prediction_inr = prediction_usd * USD_TO_INR

                st.session_state.prediction = (prediction_usd, prediction_inr)
                st.session_state.prediction_input = {
                    "Brand": brand,
                    "Year": make_year,
                    "Fuel": fuel_type,
                    "Transmission": transmission,
                    "Color": color,
                    "Mileage": f"{mileage_kmpl:.1f} kmpl",
                    "Engine": f"{engine_cc:,} cc",
                    "Owners": owner_count,
                    "Accidents": accidents_reported,
                    "Service": service_history,
                    "Insurance": insurance_valid,
                }
                st.rerun()
            except Exception as exc:
                st.error("The model could not make a prediction.")
                st.code(str(exc))
                st.info("Keep scikit-learn pinned to 1.9.0 for the supplied model.pkl.")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        prediction = st.session_state.prediction
        saved = st.session_state.prediction_input

        if prediction:
            prediction_usd, prediction_inr = prediction
            # Visual meter is intentionally a presentation element, not a confidence score.
            meter = 68

            specs_html = ""
            if saved:
                for key in ["Brand", "Year", "Fuel", "Transmission"]:
                    specs_html += (
                        f'<div class="spec"><div class="spec-label">{esc(key)}</div>'
                        f'<div class="spec-value">{esc(saved[key])}</div></div>'
                    )

            st.html(
                f"""
                <div class="result-panel">
                    <div class="result-kicker">Estimated market value</div>
                    <div class="car-graphic"><div class="car-outline"></div></div>
                    <div class="result-caption">Your vehicle is valued at</div>
                    <div class="result-price">&#8377; {prediction_inr:,.0f}</div>
                    <div class="result-usd">${prediction_usd:,.2f} USD · reference rate &#8377;{USD_TO_INR}/USD</div>

                    <div class="meter">
                        <div class="meter-head"><span>Valuation signal</span><span>Calculated</span></div>
                        <div class="meter-track"><div class="meter-fill" style="width:{meter}%"></div></div>
                    </div>

                    <div class="spec-grid">{specs_html}</div>
                    <div style="margin-top:20px;color:#8f9689;font-size:11px;line-height:1.6;">
                        Model estimate only. Actual resale value can vary with location,
                        market demand, inspection results and negotiation.
                    </div>
                </div>
                """,
)
        else:
            st.markdown(
                """
                <div class="result-panel">
                    <div class="result-kicker">Valuation preview</div>
                    <div class="car-graphic"><div class="car-outline"></div></div>
                    <div class="result-caption">Your estimate will appear here</div>
                    <div class="result-price">₹ —</div>
                    <div class="result-empty">
                        Complete the vehicle profile and calculate the value.
                        The trained pipeline will process the same feature schema
                        used during model development.
                    </div>
                    <div class="meter">
                        <div class="meter-head"><span>Prediction status</span><span>Waiting for input</span></div>
                        <div class="meter-track"><div class="meter-fill" style="width:10%"></div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# METHODOLOGY
# ============================================================
elif page == "Methodology":
    st.markdown(
        """
        <div class="eyebrow">Under the hood</div>
        <h1 class="hero-title" style="font-size:clamp(42px,5vw,68px);">
            From raw details<br>to <span>valuation.</span>
        </h1>
        <div class="hero-copy">
            The website does not rebuild the model. It sends the entered vehicle
            profile through the serialized trained pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        ("01", "Vehicle profile", "11 input signals describe the car: year, mileage, engine, fuel, ownership, brand, transmission, color, service history, accidents and insurance."),
        ("02", "Numerical processing", "The training workflow applies IQR-based clipping and feature scaling to numerical variables."),
        ("03", "Categorical encoding", "Categorical values are transformed with OneHotEncoder and unknown categories are handled by the trained pipeline."),
        ("04", "Regression", "The serialized model receives the transformed feature vector and returns a price estimate in USD."),
        ("05", "Presentation", f"The application converts the model output using the project's reference rate of ₹{USD_TO_INR} per USD."),
    ]

    for no, title, desc in cards:
        st.markdown(
            f"""
            <div class="info-card">
                <div style="color:#8ebc12;font-family:'Manrope';font-weight:800;font-size:12px;letter-spacing:1px;">{no}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# MODEL REPORT
# ============================================================
else:
    st.markdown(
        """
        <div class="eyebrow">Project evaluation</div>
        <h1 class="hero-title" style="font-size:clamp(42px,5vw,68px);">
            Model <span>report.</span>
        </h1>
        <div class="hero-copy">
            Evaluation figures below are reproduced from the supplied project README.
            They are documentation values, not metrics recalculated by the website.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)
    metric_data = [
        (a, "CV R²", "0.8705"),
        (b, "Train R²", "0.8715"),
        (c, "Test R²", "0.8736"),
        (d, "RMSE", "998.44"),
    ]

    for col, label, value in metric_data:
        with col:
            st.markdown(
                f"""
                <div class="info-card" style="min-height:120px;">
                    <div style="color:#74756d;font-size:10px;text-transform:uppercase;letter-spacing:1.4px;">{label}</div>
                    <div style="font-family:'Manrope';font-weight:800;font-size:34px;margin-top:12px;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="info-card">
            <h3>Algorithms evaluated</h3>
            <p>
                Linear Regression · Ridge Regression · Decision Tree Regressor ·
                Random Forest Regressor · Gradient Boosting Regressor ·
                K-Nearest Neighbors · XGBoost Regressor
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "The supplied project README identifies Linear Regression as the best-performing model. "
        "The serialized model artifact is kept unchanged by this deployment."
    )

st.markdown(
    """
    <div style="text-align:center;color:#8a8b82;font-size:11px;padding-top:40px;">
        CARVALUE / USED CAR PRICE INTELLIGENCE / ML DEPLOYMENT
    </div>
    """,
    unsafe_allow_html=True,
)
