import datetime
import io
import urllib3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Safe Import pytz untuk Zona Waktu WIB
try:
    import pytz
    wib_tz = pytz.timezone('Asia/Jakarta')
    now = datetime.datetime.now(wib_tz)
except Exception:
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)

# Safe Import Auto-Refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh_installed = True
except Exception:
    st_autorefresh_installed = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Smart Hydro Monitoring - S7°38.659' E113°01.641'",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto Refresh Halaman Setiap 60 Detik
if st_autorefresh_installed:
    st_autorefresh(interval=60000, limit=1000, key="datarefresh")


# Load Dataset Resmi Hydro-Oceanography & Training ML Model
@st.cache_data
def load_official_dishidros_dataset():
    raw_matrix = [
        [2.4, 2.3, 2.0, 1.6, 1.2, 0.9, 0.9, 1.0, 1.2, 1.6, 1.9, 2.1, 2.2, 2.1, 1.8, 1.5, 1.2, 1.0, 0.9, 1.0, 1.3, 1.7, 2.1, 2.3],
        [2.5, 2.4, 2.2, 1.8, 1.5, 1.1, 0.9, 0.9, 1.0, 1.2, 1.5, 1.8, 1.9, 2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.1, 1.3, 1.6, 1.9, 2.2],
        [2.4, 2.5, 2.4, 2.1, 1.8, 1.4, 1.1, 1.0, 0.9, 1.0, 1.2, 1.4, 1.6, 1.7, 1.7, 1.6, 1.5, 1.4, 1.3, 1.3, 1.4, 1.6, 1.8, 2.1],
        [2.3, 2.4, 2.4, 2.3, 2.1, 1.8, 1.5, 1.2, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.6, 1.6, 1.8, 1.9],
        [2.1, 2.2, 2.3, 2.3, 2.3, 2.1, 1.9, 1.6, 1.3, 1.1, 0.9, 0.8, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.7, 1.7, 1.7, 1.8, 1.8, 1.8],
        [1.9, 2.0, 2.1, 2.2, 2.3, 2.3, 2.2, 2.0, 1.7, 1.4, 1.1, 0.8, 0.6, 0.7, 0.9, 1.1, 1.4, 1.7, 1.9, 1.9, 1.9, 1.9, 1.9, 1.8],
        [1.8, 1.8, 1.9, 2.0, 2.2, 2.3, 2.4, 2.4, 2.1, 1.8, 1.4, 1.0, 0.6, 0.4, 0.5, 0.8, 1.2, 1.6, 1.9, 2.1, 2.1, 2.0, 1.9, 1.8],
        [1.7, 1.6, 1.6, 1.7, 1.9, 2.2, 2.4, 2.6, 2.5, 2.2, 1.8, 1.3, 0.8, 0.5, 0.2, 0.2, 0.5, 0.9, 1.4, 1.8, 2.2, 2.3, 2.2, 2.0],
        [1.8, 1.5, 1.4, 1.4, 1.6, 1.9, 2.3, 2.6, 2.7, 2.6, 2.3, 1.8, 1.2, 0.7, 0.3, 0.1, 0.2, 0.6, 1.1, 1.6, 2.1, 2.4, 2.4, 2.2],
        [1.9, 1.6, 1.3, 1.2, 1.3, 1.5, 1.9, 2.4, 2.7, 2.8, 2.6, 2.2, 1.6, 1.0, 0.5, 0.2, 0.2, 0.4, 0.8, 1.4, 1.9, 2.3, 2.5, 2.4],
        [2.1, 1.7, 1.3, 1.1, 1.0, 1.2, 1.6, 2.0, 2.5, 2.7, 2.7, 2.5, 2.0, 1.4, 0.9, 0.4, 0.3, 0.3, 0.7, 1.2, 1.7, 2.2, 2.5, 2.5],
        [2.2, 1.9, 1.4, 1.1, 0.9, 0.9, 1.2, 1.6, 2.1, 2.5, 2.7, 2.6, 2.3, 1.8, 1.3, 0.8, 0.5, 0.4, 0.6, 1.0, 1.5, 2.1, 2.4, 2.5],
        [2.4, 2.0, 1.6, 1.2, 0.9, 0.8, 0.9, 1.3, 1.7, 2.1, 2.5, 2.6, 2.4, 2.1, 1.6, 1.1, 0.8, 0.6, 0.7, 1.0, 1.4, 1.9, 2.3, 2.5],
        [2.4, 2.2, 1.8, 1.4, 1.0, 0.8, 0.8, 1.0, 1.3, 1.7, 2.1, 2.3, 2.3, 2.1, 1.8, 1.4, 1.1, 0.9, 0.9, 1.1, 1.4, 1.8, 2.1, 2.4],
        [2.4, 2.3, 2.0, 1.6, 1.2, 0.9, 0.8, 0.9, 1.1, 1.4, 1.7, 2.0, 2.1, 2.0, 1.9, 1.6, 1.4, 1.2, 1.2, 1.3, 1.5, 1.8, 2.1, 2.3],
        [2.4, 2.3, 2.1, 1.8, 1.4, 1.1, 0.9, 0.9, 1.0, 1.1, 1.4, 1.6, 1.7, 1.8, 1.8, 1.7, 1.5, 1.4, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2],
        [2.3, 2.3, 2.1, 1.9, 1.6, 1.3, 1.1, 1.0, 1.0, 1.0, 1.1, 1.3, 1.4, 1.5, 1.5, 1.6, 1.6, 1.6, 1.6, 1.7, 1.8, 1.9, 2.1, 2.2],
        [2.2, 2.2, 2.1, 2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.0, 1.0, 1.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
        [2.2, 2.2, 2.1, 2.0, 1.9, 1.8, 1.6, 1.4, 1.3, 1.1, 1.0, 1.0, 0.9, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.1, 2.1, 2.2],
        [2.1, 2.1, 2.0, 2.0, 1.9, 1.8, 1.7, 1.5, 1.3, 1.1, 1.0, 0.8, 0.7, 0.8, 0.9, 1.1, 1.4, 1.6, 1.9, 2.0, 2.1, 2.2, 2.1, 2.0],
        [2.1, 2.0, 2.0, 1.9, 2.0, 2.0, 2.0, 1.8, 1.8, 1.6, 1.3, 1.1, 0.8, 0.7, 0.6, 0.7, 0.9, 1.2, 1.5, 1.8, 2.0, 2.1, 2.1, 2.1],
        [2.0, 1.9, 1.8, 1.8, 1.9, 2.0, 2.1, 2.1, 2.0, 1.8, 1.5, 1.2, 0.9, 0.7, 0.6, 0.6, 0.8, 1.1, 1.4, 1.8, 2.0, 2.1, 2.1, 2.0],
        [1.9, 1.8, 1.7, 1.7, 1.8, 1.9, 2.1, 2.2, 2.2, 2.1, 1.8, 1.4, 1.1, 0.8, 0.6, 0.5, 0.7, 1.0, 1.4, 1.7, 2.0, 2.1, 2.1, 2.0],
        [1.8, 1.6, 1.5, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.3, 2.1, 1.7, 1.3, 0.9, 0.7, 0.5, 0.6, 0.9, 1.3, 1.7, 2.0, 2.2, 2.2, 2.0],
        [1.8, 1.5, 1.3, 1.2, 1.3, 1.5, 1.8, 2.2, 2.4, 2.4, 2.3, 1.9, 1.5, 1.1, 0.8, 0.6, 0.6, 0.9, 1.2, 1.7, 2.1, 2.3, 2.3, 2.1],
        [1.8, 1.5, 1.2, 1.0, 1.0, 1.2, 1.6, 1.9, 2.3, 2.5, 2.4, 2.2, 1.8, 1.3, 0.9, 0.7, 0.7, 0.8, 1.2, 1.6, 2.1, 2.4, 2.5, 2.3],
        [2.0, 1.5, 1.2, 0.9, 0.8, 0.9, 1.2, 1.6, 2.0, 2.4, 2.5, 2.3, 2.0, 1.6, 1.1, 0.9, 0.7, 0.9, 1.2, 1.6, 2.0, 2.4, 2.6, 2.5],
        [2.2, 1.7, 1.3, 0.9, 0.7, 0.7, 0.9, 1.3, 1.7, 2.1, 2.3, 2.3, 2.1, 1.8, 1.4, 1.0, 0.9, 0.9, 1.1, 1.5, 2.0, 2.4, 2.6, 2.6],
        [2.4, 2.0, 1.5, 1.0, 0.7, 0.6, 0.6, 0.9, 1.3, 1.8, 2.1, 2.2, 2.2, 1.9, 1.6, 1.3, 1.1, 1.0, 1.2, 1.5, 1.9, 2.3, 2.6, 2.7],
        [2.6, 2.3, 1.8, 1.3, 0.9, 0.6, 0.5, 0.7, 1.0, 1.4, 1.7, 2.0, 2.0, 2.0, 1.7, 1.5, 1.3, 1.2, 1.2, 1.4, 1.8, 2.2, 2.5, 2.7]
    ]

    records = []
    for day_idx in range(30):
        day_num = day_idx + 1
        for hour_idx in range(24):
            dt = datetime.datetime(2026, 9, day_num, hour_idx, 0, 0)
            records.append({
                'Timestamp': dt,
                'Latitude': "S7°38.659'",
                'Longitude': "E113°01.641'",
                'Day': day_num,
                'Hour': hour_idx,
                'Hour_sin': np.sin(2 * np.pi * hour_idx / 24.0),
                'Hour_cos': np.cos(2 * np.pi * hour_idx / 24.0),
                'Sea_Level_m': raw_matrix[day_idx][hour_idx]
            })

    df = pd.DataFrame(records)

    # Feature & Target Split
    X = df[['Day', 'Hour_sin', 'Hour_cos']]
    y = df['Sea_Level_m']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))

    df['ML_Predicted_Sea_Level_m'] = np.round(model.predict(X), 2)
    return df, mae, r2

df, mae_score, r2_score_val = load_official_dishidros_dataset()

# Filter Data Realtime Menurut Jam Saat Ini
current_day = now.day if now.month == 9 else 9
current_hour = now.hour

current_data = df[(df['Day'] == current_day) & (df['Hour'] == current_hour)]
realtime_level = current_data['Sea_Level_m'].values[0] if not current_data.empty else df.loc[0, 'Sea_Level_m']
ml_level = current_data['ML_Predicted_Sea_Level_m'].values[0] if not current_data.empty else df.loc[0, 'ML_Predicted_Sea_Level_m']

# Panel Kontrol Sidebar
st.sidebar.markdown("### ⚙️ Panel Kontrol EWS")
enable_audio = st.sidebar.checkbox(
    "🔔 Aktifkan Alarm Suara (EWS)", 
    value=True, 
    help="Memutar suara sirene otomatis saat level air laut < 0.2m"
)
test_alarm = st.sidebar.checkbox(
    "🧪 Tes Alarm Manual (< 0.2m)", 
    value=False, 
    help="Simulasi memicu kondisi kritis untuk menguji alarm suara"
)

# Evaluasi Kondisi Kritis (< 0.2 meter)
is_critical = (realtime_level < 0.2) or test_alarm

# =========================================================
# EARLY WARNING SYSTEM (EWS) AUDIO & VISUAL TRIGGER
# =========================================================
if is_critical:
    st.error(f"🚨 **PERINGATAN DINI CRITICAL LOW WATER LEVEL!** Level air laut berada di bawah ambang batas aman! (Level Terdeteksi: {realtime_level:.2f} m)")
    
    if enable_audio:
        # Pemicu Audio Sirene Menggunakan Web Audio API (Lintas Perangkat HP/PC)
        audio_js = """
        <script>
        function playEmergencyAlarm() {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            
            // Sirene Modulasi Darurat (850Hz ke 400Hz)
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(850, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(400, ctx.currentTime + 0.5);
            
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            
            osc.connect(gain);
            gain.connect(ctx.destination);
            
            osc.start();
            osc.stop(ctx.currentTime + 0.6);
        }
        
        // Loop Sirene Setiap 800 milidetik
        setInterval(playEmergencyAlarm, 800);
        </script>
        """
        components.html(audio_js, height=0, width=0)

# Header Utama Dashboard
st.markdown(f"""
<div style="background-color: #1e293b; border-left: 8px solid {'#ef4444' if is_critical else '#38bdf8'}; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0; font-size: 22px; color: #f8fafc;">🌊 Smart Sea Water Level Monitoring</h1>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">📍 Stasiun Intake Area PLTGU Grati — S7°38.659' E113°01.641'</p>
        </div>
        <div>
            <span style="background-color: {'#ef4444' if is_critical else '#10b981'}; color: #ffffff; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px;">
                {'🚨 CRITICAL WARNING' if is_critical else '🟢 NORMAL STATUS'}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Kartu Ringkasan Metrik
m1, m2, m3, m4 = st.columns(4)
m1.metric("Waktu Observasi", f"{now.strftime('%H:%M')} WIB")
m2.metric(
    "Sea Level Realtime", 
    f"{realtime_level:.2f} m", 
    delta="- SURUT KRITIS (<0.2m)" if is_critical else "Aman", 
    delta_color="inverse" if is_critical else "normal"
)
m3.metric("Prediksi ML Model", f"{ml_level:.2f} m")
m4.metric("Akurasi AI (R²)", f"{r2_score_val * 100:.1f}%")

# Grafik Plotly Interaktif
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Timestamp'][:24],
    y=df['Sea_Level_m'][:24],
    mode='lines+markers',
    name='Sea Level (m)',
    line=dict(color='#38bdf8', width=3),
    marker=dict(size=6)
))

# Garis Ambang Batas Kritis (Threshold Line 0.2m)
fig.add_hline(
    y=0.2, 
    line_dash="dash", 
    line_color="#ef4444", 
    annotation_text="Ambang Batas Kritis (0.2m)", 
    annotation_position="bottom right",
    annotation_font_color="#ef4444"
)

fig.update_layout(
    template='plotly_dark',
    height=380,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Waktu",
    yaxis_title="Ketinggian Air Laut (m)"
)

st.plotly_chart(fig, use_container_width=True)
