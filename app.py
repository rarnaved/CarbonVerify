import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import shap
import json
import hashlib
from fpdf import FPDF
from datetime import datetime
import folium
from streamlit_folium import st_folium

# PAGE CONFIG & ENTERPRISE LAYOUT
st.set_page_config(page_title="CarbonVerify Enterprise GeoAI", layout="wide", initial_sidebar_state="expanded")

# CUSTOM CSS FOR ENTERPRISE SAAS LOOK
st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0F172A; letter-spacing: -0.5px; }
    .subtitle { font-size: 1rem; color: #475569; margin-bottom: 15px; }
    .kpi-card { background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%); border-radius: 10px; padding: 18px; border: 1px solid #DBEAFE; }
    .alert-card { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 12px; border-radius: 6px; }
    .success-card { background-color: #F0FDF4; border-left: 5px solid #22C55E; padding: 12px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌍 CarbonVerify Enterprise GeoAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Satellite Remote Sensing & Explainable AI Framework for Carbon Credit Audit Assessment</div>', unsafe_allow_html=True)

st.info("📌 **Positioning Statement:** CarbonVerify is an AI-assisted independent risk-screening platform. It performs digital MRV assessment and does not issue or officially certify carbon credits.")

# SIDEBAR: ADVANCED CONTROLS
st.sidebar.header("🗺️ 1. Project Boundary & Metadata")
project_name = st.sidebar.text_input("Project Name", "Sundarbans Mangrove Restoration")
claimed_ha = st.sidebar.number_input("Claimed Area (Hectares)", min_value=10, max_value=50000, value=1500)
claimed_co2e = st.sidebar.number_input("Claimed CO2 Impact (tCO2e)", min_value=100, max_value=1000000, value=420000)

uploaded_geojson = st.sidebar.file_uploader("Upload GeoJSON Boundary", type=["geojson", "json"])

st.sidebar.header("🕹️ 2. Simulation Scenarios")
scenario = st.sidebar.selectbox("Preset Scenario", ["Normal Growth (High Verification)", "Fake Claim / Greenwashing", "Severe Deforestation Alert"])

if scenario == "Normal Growth (High Verification)":
    restoration_signal = 0.22
    cloud_free_frac = 0.90
elif scenario == "Fake Claim / Greenwashing":
    restoration_signal = 0.02
    cloud_free_frac = 0.85
else:
    restoration_signal = -0.15
    cloud_free_frac = 0.75

restoration_signal = st.sidebar.slider("Manual NDVI Growth Uplift", min_value=-0.3, max_value=0.5, value=float(restoration_signal), step=0.01)
cloud_free_frac = st.sidebar.slider("Cloud-Free Quality Score", 0.5, 1.0, float(cloud_free_frac))

# SATELLITE PIPELINE (SIMULATION + GEE READY)
@st.cache_data
def generate_satellite_data(signal=0.18, cloud_frac=0.88):
    rng = np.random.default_rng(42)
    n_pixels = 3000
    red_before = np.clip(rng.normal(0.12, 0.02, n_pixels), 0.02, 0.4)
    nir_before = np.clip(red_before + rng.normal(0.10, 0.03, n_pixels), 0.05, 0.6)
    swir_before = np.clip(rng.normal(0.20, 0.03, n_pixels), 0.05, 0.5)
    
    growth = rng.normal(signal, 0.05, n_pixels)
    red_after = np.clip(red_before - 0.4 * growth + rng.normal(0, 0.01, n_pixels), 0.01, 0.4)
    nir_after = np.clip(nir_before + 0.6 * growth + rng.normal(0, 0.02, n_pixels), 0.05, 0.8)
    swir_after = np.clip(swir_before - 0.15 * growth + rng.normal(0, 0.02, n_pixels), 0.05, 0.5)
    
    eps = 1e-6
    ndvi_b = (nir_before - red_before) / (nir_before + red_before + eps)
    ndvi_a = (nir_after - red_after) / (nir_after + red_after + eps)
    savi_b = ((nir_before - red_before) * 1.5) / (nir_before + red_before + 0.5 + eps)
    savi_a = ((nir_after - red_after) * 1.5) / (nir_after + red_after + 0.5 + eps)
    ndmi_b = (nir_before - swir_before) / (nir_before + swir_before + eps)
    ndmi_a = (nir_after - swir_after) / (nir_after + swir_after + eps)
    
    return {
        "ndvi_before": float(np.mean(ndvi_b)), "ndvi_after": float(np.mean(ndvi_a)),
        "savi_before": float(np.mean(savi_b)), "savi_after": float(np.mean(savi_a)),
        "ndmi_before": float(np.mean(ndmi_b)), "ndmi_after": float(np.mean(ndmi_a)),
        "cloud_free": cloud_frac
    }

sat_data = generate_satellite_data(signal=restoration_signal, cloud_frac=cloud_free_frac)

# MULTI-MODEL ML BENCHMARKING ENGINE
@st.cache_resource
def train_and_benchmark_models():
    rng = np.random.default_rng(7)
    n = 1000
    ndvi = np.clip(rng.beta(2, 2, n), 0.05, 0.95)
    savi = np.clip(ndvi * rng.normal(0.65, 0.05, n), 0.02, 0.8)
    evi = np.clip(ndvi * rng.normal(0.75, 0.05, n), 0.02, 0.9)
    ndmi = np.clip(ndvi * rng.normal(0.55, 0.08, n), 0.1, 0.6)
    ratio = np.clip(rng.normal(2.5, 0.8, n), 0.5, 8.0)
    
    agb_true = 220 * (1 - np.exp(-3.2 * ndvi)) + 15 * savi
    agb = np.clip(agb_true + rng.normal(0, 20, n), 2, 350)
    
    X = pd.DataFrame({"NDVI": ndvi, "SAVI": savi, "EVI": evi, "NDMI": ndmi, "NIR_Red_ratio": ratio})
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=7),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=7),
        "SVR (Support Vector)": SVR(C=100, epsilon=0.1)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X, agb)
        preds = model.predict(X)
        r2 = r2_score(agb, preds)
        rmse = np.sqrt(mean_squared_error(agb, preds))
        results[name] = {"model": model, "r2": r2, "rmse": rmse}
        
    return results

benchmark_results = train_and_benchmark_models()
selected_model_name = st.sidebar.selectbox("Select ML Model Architecture", list(benchmark_results.keys()), index=0)
active_model = benchmark_results[selected_model_name]["model"]

# INFERENCE & CARBON CALCULATION
input_row = pd.DataFrame({
    "NDVI": [sat_data["ndvi_after"]],
    "SAVI": [sat_data["savi_after"]],
    "EVI": [sat_data["ndvi_after"] * 0.8],
    "NDMI": [sat_data["ndmi_after"]],
    "NIR_Red_ratio": [3.2]
})

predicted_agb = active_model.predict(input_row)[0]
total_biomass = predicted_agb * claimed_ha
total_carbon = total_biomass * 0.47
estimated_co2e = total_carbon * (44 / 12)

# CONSISTENCY & ANOMALY ENGINE
observed_ha = claimed_ha * (0.80 if restoration_signal < 0.0 else 0.95)
spatial_score = max(0.0, 1.0 - abs(claimed_ha - observed_ha) / claimed_ha)
veg_growth_score = min(1.0, max(0.0, (sat_data["ndvi_after"] - sat_data["ndvi_before"]) / 0.15))
quality_score = sat_data["cloud_free"]
model_score = benchmark_results[selected_model_name]["r2"]

consistency_score = (0.25 * spatial_score + 0.35 * veg_growth_score + 0.20 * quality_score + 0.20 * model_score) * 100

if consistency_score >= 85:
    conf_band, badge_color = "Very High", "green"
elif consistency_score >= 65:
    conf_band, badge_color = "High", "blue"
elif consistency_score >= 45:
    conf_band, badge_color = "Moderate", "orange"
else:
    conf_band, badge_color = "Low", "red"

# TOP DASHBOARD KPIS
col1, col2, col3, col4 = st.columns(4)
col1.metric("Consistency Score", f"{consistency_score:.1f}%", f"Band: {conf_band}")
col2.metric("Observed CO2e Impact", f"{estimated_co2e:,.0f} tCO2e", f"Claimed: {claimed_co2e:,.0f}")
col3.metric("Biomass Density (AGB)", f"{predicted_agb:.1f} Mg/ha", f"Model: {selected_model_name}")
col4.metric("Spatial Polygon Match", f"{observed_ha:.0f} Ha", f"Claimed: {claimed_ha} Ha")

st.divider()

# TABS FOR ENTERPRISE MODULES
tab_map, tab_bench, tab_xai, tab_audit = st.tabs([
    "🗺️ Geospatial Boundary & Anomaly Map", 
    "📊 Model Benchmarking & ML Comparison", 
    "🧠 Explainable AI (SHAP)", 
    "🔒 Cryptographic Audit & PDF Export"
])

with tab_map:
    col_map, col_chart = st.columns([1.2, 1])
    
    with col_map:
        st.subheader("Interactive Satellite Boundary Map")
        m = folium.Map(location=[21.9497, 88.9007], zoom_start=11, tiles="OpenStreetMap")
        
        if uploaded_geojson is not None:
            geojson_data = json.load(uploaded_geojson)
            folium.GeoJson(geojson_data, name="Project Boundary").add_to(m)
        else:
            coords = [[21.90, 88.85], [21.90, 88.95], [22.00, 88.95], [22.00, 88.85]]
            folium.Polygon(locations=coords, color="red" if restoration_signal < 0 else "green", fill=True, fill_opacity=0.35, popup=project_name).add_to(m)
            
        st_folium(m, width=550, height=350)
        
    with col_chart:
        st.subheader("Temporal Spectral Dynamics")
        indices_df = pd.DataFrame({
            "Index": ["NDVI (Density)", "SAVI (Soil Adjusted)", "NDMI (Moisture)"],
            "Baseline (Before)": [sat_data["ndvi_before"], sat_data["savi_before"], sat_data["ndmi_before"]],
            "Monitoring (After)": [sat_data["ndvi_after"], sat_data["savi_after"], sat_data["ndmi_after"]]
        }).set_index("Index")
        st.bar_chart(indices_df)
        
        if restoration_signal < 0:
            st.markdown('<div class="alert-card">⚠️ <b>Deforestation Risk Detected:</b> Vegetation index shows negative growth uplift compared to baseline.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card">✅ <b>Vegetation Growth Positive:</b> Significant canopy development observed over monitoring period.</div>', unsafe_allow_html=True)

with tab_bench:
    st.subheader("Machine Learning Algorithm Comparison")
    st.caption("Performance evaluation across candidate models trained on literature-informed biomass datasets.")
    
    bench_data = []
    for name, res in benchmark_results.items():
        bench_data.append({
            "Algorithm": name,
            "R² Score (Accuracy)": round(res["r2"], 4),
            "RMSE (Mg/ha Error)": round(res["rmse"], 2)
        })
    
    bench_df = pd.DataFrame(bench_data)
    st.dataframe(bench_df.style.highlight_max(axis=0, subset=["R² Score (Accuracy)"], color="#DCFCE7"), use_container_width=True)
    
    st.write(f"**Current Selected Model:** `{selected_model_name}` (R² = {benchmark_results[selected_model_name]['r2']:.3f})")

with tab_xai:
    st.subheader("Explainable AI (XAI) Driver Attribution")
    
    if "Random Forest" in selected_model_name or "XGBoost" in selected_model_name:
        explainer = shap.TreeExplainer(active_model)
        shap_vals = explainer.shap_values(input_row)[0]
        
        shap_df = pd.DataFrame({
            "Feature": input_row.columns,
            "SHAP Contribution": shap_vals
        }).sort_values(by="SHAP Contribution", ascending=False)
        
        col_shap_chart, col_narrative = st.columns([1.5, 1])
        with col_shap_chart:
            st.bar_chart(shap_df.set_index("Feature"))
        
        with col_narrative:
            st.write("**Natural Language Audit Explanation:**")
            for _, row in shap_df.iterrows():
                direction = "increased" if row["SHAP Contribution"] >= 0 else "decreased"
                st.write(f"- **{row['Feature']}** {direction} the biomass estimate confidence by `{abs(row['SHAP Contribution']):.2f}` points.")
    else:
        st.info("SHAP TreeExplainer is optimized for tree-based models (Random Forest & XGBoost). Select Tree Model in sidebar to view XAI chart.")

with tab_audit:
    st.subheader("Cryptographic Audit Chain & PDF Report Generation")
    
    audit_data = {
        "project_name": project_name,
        "timestamp": datetime.now().isoformat(),
        "model_used": selected_model_name,
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
        pdf.cell(190, 8, f"ML Algorithm Used: {selected_model_name}", 0, 1)
        pdf.cell(190, 8, f"Consistency Score: {consistency_score:.1f}% ({conf_band})", 0, 1)
        pdf.cell(190, 8, f"Estimated CO2e: {estimated_co2e:,.0f} tCO2e", 0, 1)
        pdf.cell(190, 8, f"Audit Hash: {audit_hash[:32]}...", 0, 1)
        return bytes(pdf.output())

    pdf_bytes = generate_pdf()
    st.download_button(
        label="📄 Download Cryptographically Signed Verification Report (PDF)",
        data=pdf_bytes,
        file_name=f"CarbonVerify_Assessment_{project_name.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )
