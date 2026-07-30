import requests
import streamlit as st

from fleetsense.features.data_loader import load_schema

st.set_page_config(page_title="FleetSense", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
    /* ---- Page base ---- */
    .stApp {
        background-color: #EEF2F6;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* ---- Header banner ---- */
    .fleetsense-header {
        background: linear-gradient(90deg, #001f3d 0%, #13315C 100%);
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .fleetsense-header::after {
        content: "";
        position: absolute;
        right: -20px; top: -20px;
        width: 140px; height: 140px;
        border-radius: 50%;
        background: #E8590C22;
    }
    .fleetsense-title {
        color: #fff;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .fleetsense-subtitle {
        color: #9FC2E0;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }

    /* ---- Form container ---- */
    div[data-testid="stForm"] {
        background-color: #fff;
        border: 1px solid #D8E0E8;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 1px 3px rgba(0,31,61,0.06);
    }

    /* ---- Section labels (expander headers used as group titles) ---- */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #001f3d;
        background-color: #F4F7FA;
        border-radius: 8px;
    }

    /* ---- Compact number inputs: gauge-readout style ---- */
    div[data-testid="stNumberInput"] {
        max-width: 140px;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.78rem;
        color: #4A5C6E;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    div[data-testid="stNumberInput"] input {
        font-family: "SF Mono", "Consolas", monospace;
        font-size: 0.85rem;
        padding: 0.35rem 0.5rem;
        border: 1px solid #C7D2DC;
        border-radius: 6px;
        background-color: #FAFBFC;
        color: #001f3d;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #E8590C;
        box-shadow: 0 0 0 1px #E8590C33;
    }

    /* ---- Submit button: distress-orange, unmissable ---- */
    .stFormSubmitButton button {
        background-color: #E8590C;
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        margin-top: 0.5rem;
    }
    .stFormSubmitButton button:hover {
        background-color: #C74A08;
        color: #fff;
    }

    /* ---- Result card ---- */
    .result-card {
        background: linear-gradient(135deg, #001f3d 0%, #13315C 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        border-left: 5px solid #E8590C;
    }
    .result-label {
        color: #9FC2E0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .result-value {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="fleetsense-header">
    <p class="fleetsense-title">⚓ FleetSense</p>
    <p class="fleetsense-subtitle">Vessel type classification from AIS feature data</p>
</div>
""",
    unsafe_allow_html=True,
)

SCHEMA = load_schema()
FEATURES = SCHEMA["columns"]


with st.form("feature_input", clear_on_submit=True, enter_to_submit=True, border=True):
    st.write("Input vessel features")
    values = {}
    cols = st.columns(3)
    for i, feature in enumerate(FEATURES):
        feature_name = str(feature).replace("_", " ")
        feature_name = feature_name[0].upper() + feature_name[1:]
        with cols[i % 3]:
            values[feature] = st.number_input(feature_name, key=feature)

    submitted = st.form_submit_button("Predict")

if submitted:
    response = requests.post("http://localhost:8000/predict", json=values)
    result = response.json()

    st.markdown(
        f"""
    <div class="result-card">
        <div class="result-label">Predicted vessel type</div>
        <div class="result-value">{result["vessel_type"]}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.bar_chart(result["probabilities"])
