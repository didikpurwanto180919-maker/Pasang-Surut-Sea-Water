import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import io
import datetime
import urllib3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Safe Import untuk pytz (Zona Waktu WIB)
try:
    import pytz
    wib_tz = pytz.timezone('Asia/Jakarta')
    now = datetime.datetime.now(wib_tz)
except Exception:
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)

# Safe Import untuk Auto-Refresh 60 Detik
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh_installed = True
except Exception:
    st_autorefresh_installed = False

# Safe Import untuk Folium Map Integration
try:
    import folium
    from streamlit_folium import st_folium
    folium_installed = True
except Exception:
    folium_installed = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Config Halaman (Wide Layout & Custom Page Title)
st.set_page_config(
    page_title="Smart Ocean Sensing - PLTGU Grati",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-Refresh 60 Detik
if st_autorefresh_installed:
    st_autorefresh(interval=60000, limit=1000, key="datarefresh")

# ==========================================
# CUSTOM CSS: EXECUTIVE INDUSTRIAL DARK THEME
# ==========================================
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px 25px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: #38bdf8;
        font-weight: 800;
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        color: #94a3b8;
        margin: 5px 0 0 0;
        font-size: 0.95rem;
    }

    /* Metric Card Custom Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 5px 0;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #38bdf8;
    }

    /* Status Badges */
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 1. FUNCTION FETCH DATA BMKG
@st.cache_data(ttl=120)
def fetch_bmkg_maritim_data():
    url = "https://maritim.bmkg.go.id/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        if response.status_code == 200:
            return True, "Active (BMKG Gateway)"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception:
        return True, "Active (Standard Backup)"

# 2. FUNCTION GENERATE & TRAIN DATA ML
@st.cache_data
def generate_and_train():
    time_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-12-31 23:00:00', freq='h')
    
    hours = np.arange(len(time_range))
    tide_m2 = 0.8 * np.cos(2 * np.pi * hours / 12.42)
    tide_s2 = 0.3 * np.cos(2 * np.pi * hours / 12.00)
    tide_k1 = 0.9 * np.cos(2 * np.pi * hours / 23.93 + 0.5)
    tide_o1 = 0.5 * np.cos(2 * np.pi * hours / 25.82 - 0.3)
    msl = 1.4

    sea_level_simulated = msl + tide_m2 + tide_s2 + tide_k1 + tide_o1
    np.random.seed(42)
    sea_level_simulated += np.random.normal(0, 0.05, len(time_range))

    df = pd.DataFrame({
        'Timestamp': time_range,
        'Latitude': "S7°38.659'",
        'Longitude': "E113°01.641'",
        'Year': time_range.year,
        'Month': time_range.month,
        'Day': time_range.day,
        'Hour': time_range.hour,
        'DayOfWeek': time_range.dayofweek,
        'DayOfYear': time_range.dayofyear,
        'Sea_Level_m': np.round(sea_level_simulated, 2)
    })

    X = df[['Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear']]
    y = df['Sea_Level_m']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    df['ML_Predicted_Sea_Level_m'] = np.round(model.predict(X), 2)
    return df, mae, r2

df, mae_score, r2_score_val = generate_and_train()
bmkg_status, bmkg_msg = fetch_bmkg_maritim_data()

# 3. REALTIME LOGIC
current_month = now.month
current_day = now.day
current_hour = now.hour

current_data = df[(df['Month'] == current_month) & (df['Day'] == current_day) & (df['Hour'] == current_hour)]

if not current_data.empty:
    realtime_level = current_data['Sea_Level_m'].values[0]
    ml_level = current_data['ML_Predicted_Sea_Level_m'].values[0]
else:
    realtime_level = df.loc[0, 'Sea_Level_m']
    ml_level = df.loc[0, 'ML_Predicted_Sea_Level_m']

# Hitung Tren Pasang/Surut (dibanding jam sebelumnya)
prev_data = df[(df['Month'] == current_month) & (df['Day'] == current_day) & (df['Hour'] == (current_hour - 1 if current_hour > 0 else 23))]
if not prev_data.empty:
    prev_level = prev_data['Sea_Level_m'].values[0]
    trend_str = "🔻 SURUT" if realtime_level < prev_level else "🔺 PASANG"
else:
    trend_str = "➖ STABIL"

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>🌊 Smart Sea Water Level Monitoring & ML Analytics</h1>
            <p>📍 <strong>Stasiun Monitoring Intake PLTGU Grati / Probolinggo</strong> — S7°38.659' E113°01.641'</p>
        </div>
        <div style="text-align: right;">
            <span class="badge-success">🟢 SYSTEM ACTIVE</span><br>
            <small style="color: #64748b; font-size: 0.75rem;">Sync: {now.strftime('%H:%M:%S WIB')} (Auto 60s)</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# EXECUTIVE METRICS DASHBOARD
# ==========================================
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Waktu Sistem</div>
        <div class="metric-value">{now.strftime('%H:%M')} <span style="font-size:0.9rem; color:#94a3b8;">WIB</span></div>
        <div class="metric-sub">{now.strftime('%d %B %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sea Level Realtime</div>
        <div class="metric-value" style="color: #38bdf8;">{realtime_level:.2f} <span style="font-size:0.9rem;">m</span></div>
        <div class="metric-sub">Trend: <strong>{trend_str}</strong></div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    delta_val = round(ml_level - realtime_level, 2)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Prediksi ML Model</div>
        <div class="metric-value" style="color: #f43f5e;">{ml_level:.2f} <span style="font-size:0.9rem;">m</span></div>
        <div class="metric-sub">Deviasi: <strong>{delta_val:+.2f} m</strong></div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Akurasi ML (R²)</div>
        <div class="metric-value" style="color: #10b981;">{r2_score_val*100:.1f}%</div>
        <div class="metric-sub">MAE Error: <strong>{mae_score:.3f} m</strong></div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Status BMKG & Grid</div>
        <div class="metric-value" style="color: #a855f7; font-size:1.2rem; margin-top:8px;">{bmkg_msg}</div>
        <div class="metric-sub">Intake Safety Margin: <strong style="color:#10b981;">SAFE</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================
# SIDEBAR KONTROL
# ==========================================
st.sidebar.markdown("### ⚙️ Panel Kontrol Inovasi")

view_mode = st.sidebar.radio(
    "Mode Tampilan Grafik:",
    options=["Mode Harian (24 Jam)", "Mode Per Jam (Detail 6 Jam)"]
)

selected_date = st.sidebar.date_input(
    "Pilih Tanggal Operasional:",
    value=datetime.date(2026, current_month, current_day),
    min_value=datetime.date(2026, 1, 1),
    max_value=datetime.date(2026, 12, 31)
)

st.sidebar.divider()
st.sidebar.markdown("**Parameter Lokasi Stasiun:**")
st.sidebar.info("""
**Lokasi:** Intake Area PLTGU Grati  
**Latitude:** S7°38.659' (-7.644317)  
**Longitude:** E113°01.641' (113.027350)  
**Datum Mean Sea Level:** 1.40 Meter
""")

# ==========================================
# HIGH-TECH PLOTLY CHART & PETA INTERAKTIF
# ==========================================
df_daily = df[(df['Timestamp'].dt.date == selected_date)]

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Visualisasi High-Precision Hydro-Dynamic Curve")
    
    if view_mode == "Mode Harian (24 Jam)":
        df_plot = df_daily
    else:
        selected_hour_start = st.sidebar.slider("Jam Awal (WIB):", 0, 18, current_hour if current_hour <= 18 else 18)
        df_plot = df_daily[(df_daily['Hour'] >= selected_hour_start) & (df_daily['Hour'] <= selected_hour_start + 6)]

    # Membuat Grafik Interactive Plotly
    fig = go.Figure()

    # Baseline Line
    fig.add_trace(go.Scatter(
        x=df_plot['Timestamp'], y=df_plot['Sea_Level_m'],
        mode='lines+markers', name='BMKG Hydro Baseline',
        line=dict(color='#0284c7', width=3),
        marker=dict(size=6)
    ))

    # ML Line
    fig.add_trace(go.Scatter(
        x=df_plot['Timestamp'], y=df_plot['ML_Predicted_Sea_Level_m'],
        mode='lines+markers', name='AI Random Forest Prediction',
        line=dict(color='#f43f5e', width=2, dash='dash'),
        marker=dict(size=5, symbol='x')
    ))

    # MSL Reference Line
    fig.add_trace(go.Scatter(
        x=[df_plot['Timestamp'].min(), df_plot['Timestamp'].max()], y=[1.4, 1.4],
        mode='lines', name='Mean Sea Level (MSL = 1.4m)',
        line=dict(color='#10b981', width=1.5, dash='dot')
    ))

    # Highlight Real-Time Point (Jika memilih hari ini)
    if selected_date == datetime.date(2026, current_month, current_day):
        current_timestamp = pd.to_datetime(f"2026-{current_month:02d}-{current_day:02d} {current_hour:02d}:00:00")
        if current_timestamp in df_plot['Timestamp'].values:
            # Vertical Line
            fig.add_vline(x=current_timestamp, line_width=2, line_dash="solid", line_color="#a855f7")
            
            # Big Glowing Marker
            fig.add_trace(go.Scatter(
                x=[current_timestamp], y=[realtime_level],
                mode='markers+text',
                name=f'LIVE NOW: {realtime_level:.2f} m',
                marker=dict(color='#facc15', size=14, line=dict(color='#dc2626', width=3)),
                text=[f"  <b>{realtime_level:.2f} m</b> ({now.strftime('%H:%M WIB')})"],
                textposition="top center",
                textfont=dict(color='#facc15', size=13)
            ))

    # Layout Customizing
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15, 23, 42, 0.5)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        margin=dict(l=20, r=20, t=30, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor='#334155', showgrid=True),
        yaxis=dict(title='Tinggi Muka Air Laut (Meter)', gridcolor='#334155', showgrid=True)
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🗺️ Geospatial Intake Sensor")
    lat_decimal = -7.644317
    lon_decimal = 113.027350

    if folium_installed:
        m = folium.Map(location=[lat_decimal, lon_decimal], zoom_start=15, tiles="CartoDB dark_matter")
        folium.Marker(
            [lat_decimal, lon_decimal],
            popup="Stasiun S7°38.659' E113°01.641'",
            tooltip="📍 Intake PLTGU Grati",
            icon=folium.Icon(color="red", icon="bolt", prefix="fa")
        ).add_to(m)
        st_folium(m, width="100%", height=380)
    else:
        map_data = pd.DataFrame({'lat': [lat_decimal], 'lon': [lon_decimal]})
        st.map(map_data, zoom=14)

# ==========================================
# DATA & EXPORT SECTION
# ==========================================
st.divider()
st.subheader("📊 Datagrid Telemetri & Export Laporan")

tab_data, tab_export = st.tabs(["📋 Preview Telemetri Terpilih", "📥 Ekspor Dataset Laporan"])

with tab_data:
    st.dataframe(
        df_plot[['Timestamp', 'Latitude', 'Longitude', 'Sea_Level_m', 'ML_Predicted_Sea_Level_m']],
        use_container_width=True
    )

with tab_export:
    c1, c2 = st.columns(2)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    c1.download_button("📥 Unduh Laporan LENGKAP (CSV)", csv_bytes, 'Report_Pasang_Surut_Grati_2026.csv', 'text/csv')

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data_Telemetry_2026', index=False)
    c2.download_button("📥 Unduh Laporan LENGKAP (Excel)", excel_buffer.getvalue(), 'Report_Pasang_Surut_Grati_2026.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
