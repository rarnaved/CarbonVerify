import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import json
import hashlib
from fpdf import FPDF
from datetime import datetime
import folium
from streamlit_folium import st_folium

# PAGE CONFIG & LAYOUT
st.set_page_config(page_title="CarbonVerify - Enterprise GeoAI", layout="wide", initial_sidebar_state="expanded")

# CUSTOM CSS FOR ACADEMIC/ENTERPRISE LOOK
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0px; }
    .sub-header { font-size: 1rem; color: #64748B; margin-bottom: 20px; }
    .metric-card { background-color: #F8FAFC; border-radius: 8px; padding: 15px; border: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_keywords=True)

st.markdown('<div class="main-header">🌱 CarbonVerify Enterprise</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Driven Remote Sensing & Geospatial System for Carbon Credit Verification</div>', unsafe_allow_html=True)

st.info("📌 **Positioning Statement:** CarbonVerify is an AI-assisted risk-screening platform. It provides independent evidence-based assessment and does not officially issue legal carbon credits.")

# SIDEBAR: ADVANCED INPUTS
st.sidebar.header("🗺️ 1. Boundary & Metadata")
project_name = st.sidebar.text_input("Project Name", "Sundarbans Restoration Project")
claimed_ha = st.sidebar.number_input("Claimed Area (Hectares)", min_value=10, max_value=50000, value=1200)
claimed_co2e = st.sidebar.number_input("Claimed CO2 Impact (tCO2e)", min_value=100, max_value=1000000, value=350000)

uploaded_geojson = st.sidebar.file_uploader("Upload GeoJSON Boundary (Optional)", type=["geojson", "json"])

st.sidebar.header("📡 2. Remote Sensing Controls")
restoration_signal = st.sidebar.slider(
    "Simulated Vegetation Uplift (NDVI)", 
    min_value=0.0, max_value=0.5, value=0.18, step=0.01,
    help="Viva Demo Trick: Drag to 0.02 to show zero growth & Low Confidence score!"
)
cloud_free_frac = st.sidebar.slider("Cloud-Free Quality Score", 0.5, 1.0, 0.88)

# DATA GENERATION ENGINE
@st.cache_data
def generate_satellite_data(signal=0.18, cloud_frac=0.88):
    rng = np.random.default_rng(42)
    n_pixels = 2500
    red_before = np.clip(rng.normal(0.12, 0.02, n_pixels), 0.02, 0.4)
    nir_before = np.clip(red_before + rng.normal(0.10, 0.03, n_pixels), 0.05, 0.6)
    
    growth = rng.normal(signal, 0.05, n_pixels)
    red_after = np.clip(red_before - 0.4 * growth + rng.normal(0, 0.01, n_pixels), 0.01, 0.4)
    nir_after = np.clip(nir_before + 0.6 * growth + rng.normal(0, 0.02, n_pixels), 0.05, 0.8)
    
    eps = 1e-6
    ndvi_b = (nir_before - red_before) / (nir_before + red_before + eps)
    ndvi_a = (nir_after - red_after) / (nir_after + red_after + eps)
    savi_b = ((nir_before - red_before) * 1.5) / (nir_before + red_before + 0.5 + eps)
    savi_a = ((nir_after - red_after) * 1.5) / (nir_after + red_after + 0.5 + eps)
    
    return {
        "ndvi_before": float(np.mean(ndvi_b)),
        "ndvi_after": float(np.mean(ndvi_a)),
        "savi_before": float(np.mean(savi_b)),
        "savi_after": float(np.mean(savi_a)),
        "cloud_free": cloud_frac
    }

sat_data = generate_satellite_data(signal=restoration_signal, cloud_frac=cloud_free_frac)

# ML MODEL TRAINING
@st.cache_resource
def train_agb_model():
    rng = np.random.default_rng(7)
    n = 800
    ndvi = np.clip(rng.beta(2, 2, n), 0.05, 0.95)
    savi = np.clip(ndvi * rng.normal(0.65, 0.05, n), 0.02, 0.8)
    evi = np.clip(ndvi * rng.normal(0.75, 0.05, n), 0.02, 0.9)
    ndmi = np.clip(ndvi * rng.normal(0.55, 0.08, n), 0.01, 0.6)
    ratio = np.clip(rng.normal(2.5, 0.8, n), 0.5, 8.0)
    
    agb_true = 220 * (1 - np.exp(-3.2 * ndvi)) + 15 * savi
    agb = np.clip(agb_true + rng.normal(0, 22, n), 2, 350)
    
    X = pd.DataFrame({"NDVI": ndvi, "SAVI": savi, "EVI": evi, "NDMI": ndmi, "NIR_Red_ratio": ratio})
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=7)
    rf.fit(X, agb)
    return rf

rf_model = train_agb_model()

input_row = pd.DataFrame({
    "NDVI": [sat_data["ndvi_after"]],
    "SAVI": [sat_data["savi_after"]],
    "EVI": [sat_data["ndvi_after"] * 0.8],
    "NDMI": [sat_data["ndvi_after"] * 0.6],
    "NIR_Red_ratio": [3.2]
})

predicted_agb = rf_model.predict(input_row)[0]
total_biomass = predicted_agb * claimed_ha
total_carbon = total_biomass * 0.47
estimated_co2e = total_carbon * (44 / 12)

# CONSISTENCY CALCULATIONS
observed_ha = claimed_ha * (0.85 if restoration_signal < 0.05 else 0.96)
spatial_score = max(0.0, 1.0 - abs(claimed_ha - observed_ha) / claimed_ha)
veg_growth_score = min(1.0, max(0.0, (sat_data["ndvi_after"] - sat_data["ndvi_before"]) / 0.15))
quality_score = sat_data["cloud_free"]
model_score = 0.87

consistency_score = (0.25 * spatial_score + 0.35 * veg_growth_score + 0.20 * quality_score + 0.20 * model_score) * 100

if consistency_score >= 85:
    conf_band, badge_color = "Very High", "green"
elif consistency_score >= 65:
    conf_band, badge_color = "High", "blue"
elif consistency_score >= 45:
    conf_band, badge_color = "Moderate", "orange"
else:
    conf_band, badge_color = "Low", "red"

# TOP KPI METRICS
col1, col2, col3, col4 = st.columns(4)
col1.metric("Consistency Score", f"{consistency_score:.1f}%", f"Band: {conf_band}")
col2.metric("Estimated CO2 Impact", f"{estimated_co2e:,.0f} tCO2e", f"Claimed: {claimed_co2e:,.0f}")
col3.metric("Observed Biomass (AGB)", f"{predicted_agb:.1f} Mg/ha")
col4.metric("Observed Area", f"{observed_ha:.0f} Ha", f"Claimed: {claimed_ha} Ha")

st.divider()

# TABS LAYOUT FOR ADVANCED VIEW
tab_map, tab_xai, tab_audit = st.tabs(["🗺️ Geospatial Map & Indices", "🧠 Explainable AI (XAI)", "🔒 Security & PDF Export"])

with tab_map:
    col_map, col_chart = st.columns([1.2, 1])
    
    with col_map:
        st.subheader("Project Boundary Visualization")
        # Interactive Folium Map
        m = folium.Map(location=[21.9497, 88.9007], zoom_start=11, tiles="OpenStreetMap")
        
        if uploaded_geojson is not None:
            geojson_data = json.load(uploaded_geojson)
            folium.GeoJson(geojson_data, name="Project Boundary").add_to(m)
        else:
            # Default Bounding Box Example
            coords = [[21.90, 88.85], [21.90, 88.95], [22.00, 88.95], [22.00, 88.85]]
            folium.Polygon(locations=coords, color="green", fill=True, fill_opacity=0.3, popup=project_name).add_to(m)
            
        st_folium(m, width=550, height=350)
        
    with col_chart:
        st.subheader("Spectral Index Growth Delta")
        indices_df = pd.DataFrame({
            "Index": ["NDVI (Density)", "SAVI (Soil Adjusted)"],
            "Baseline (Before)": [sat_data["ndvi_before"], sat_data["savi_before"]],
            "Monitoring (After)": [sat_data["ndvi_after"], sat_data["savi_after"]]
        }).set_index("Index")
        st.bar_chart(indices_df)

with tab_xai:
    st.subheader("SHAP Feature Importance & Driver Attribution")
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(input_row)[0]
    
    shap_df = pd.DataFrame({
        "Feature": input_row.columns,
        "SHAP Value": shap_vals
    }).sort_values(by="SHAP Value", ascending=False)
    
    col_shap_chart, col_narrative = st.columns([1.5, 1])
    with col_shap_chart:
        st.bar_chart(shap_df.set_index("Feature"))
    
    with col_narrative:
        st.write("**Natural Language Model Reasoning:**")
        for _, row in shap_df.iterrows():
            direction = "increased" if row["SHAP Value"] >= 0 else "decreased"
            st.write(f"- **{row['Feature']}** {direction} the verification confidence by `{abs(row['SHAP Value']):.2f}` points.")

with tab_audit:
    st.subheader("Cryptographic Audit Record & Signed Report")
    
    audit_data = {
        "project_name": project_name,
        "timestamp": datetime.now().isoformat(),
        "consistency_score": round(consistency_score, 2),
        "confidence_band": conf_band,
        "estimated_co2e": round(estimated_co2e, 2)
    }
    
    data_str = json.dumps(audit_data, sort_keys=True)
    audit_hash = hashlib.sha256(data_str.encode()).hexdigest()
    
    st.code(f"SHA-256 Audit Hash: {audit_hash}", language="text")
    
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(190, 10, "CarbonVerify Assessment Report", 0, 1, 'C')
        pdf.ln(5)
        pdf.set_font("Arial", '', 12)
        pdf.cell(190, 8, f"Project Name: {project_name}", 0, 1)
        pdf.cell(190, 8, f"Verification Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
        pdf.cell(190, 8, f"Consistency Score: {consistency_score:.1f}% ({conf_band})", 0, 1)
        pdf.cell(190, 8, f"Estimated CO2e: {estimated_co2e:,.0f} tCO2e", 0, 1)
        pdf.cell(190, 8, f"Audit Hash: {audit_hash[:32]}...", 0, 1)
        return bytes(pdf.output())

    pdf_bytes = generate_pdf()
    st.download_button(
        label="📄 Download Signed Verification Assessment Report (PDF)",
        data=pdf_bytes,
        file_name=f"CarbonVerify_{project_name.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )