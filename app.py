import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import datetime
import urllib3
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Safe Import pytz
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

# Config Halaman
st.set_page_config(
    page_title="Smart Sea Water Level Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if st_autorefresh_installed:
    st_autorefresh(interval=60000, limit=1000, key="datarefresh")

# ==========================================
# CUSTOM CSS: DESAIN ALARM & DASHBOARD
# ==========================================
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 18px !important; }
    .stApp { background-color: #0b0f19; color: #e2e8f0; }

    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 25px 30px;
        border-radius: 14px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 25px;
    }
    .main-header h1 { color: #38bdf8; font-weight: 800; margin: 0; font-size: 2.1rem !important; }
    .main-header p { color: #cbd5e1; margin: 8px 0 0 0; font-size: 1.1rem !important; }

    .metric-card {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }
    .metric-title { font-size: 0.95rem !important; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.8px; }
    .metric-value { font-size: 2.1rem !important; font-weight: 800; color: #f8fafc; margin: 8px 0; }
    .metric-sub { font-size: 0.95rem !important; color: #38bdf8; font-weight: 600; }
    
    .badge-success { background-color: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid #10b981; padding: 4px 12px; border-radius: 8px; font-size: 0.95rem !important; font-weight: 700; }

    /* Animation Kedip Merah Alarm */
    @keyframes blink {
        0% { background-color: #7f1d1d; opacity: 1; }
        50% { background-color: #dc2626; opacity: 0.7; }
        100% { background-color: #7f1d1d; opacity: 1; }
    }
    .alarm-banner {
        animation: blink 1s infinite;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 20px;
        border: 2px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# LOAD DATASET & MACHINE LEARNING
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
        [2.1, 2.0, 2.0, 1.9, 2.0, 2.0, 2.0, 1.9, 1.8, 1.6, 1.3, 1.1, 0.8, 0.7, 0.6, 0.7, 0.9, 1.2, 1.5, 1.8, 2.0, 2.1, 2.1, 2.1],
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
            val = raw_matrix[day_idx][hour_idx]
            dt = datetime.datetime(2026, 9, day_num, hour_idx, 0, 0)
            records.append({
                'Timestamp': dt,
                'Latitude': "S7°38.659'",
                'Longitude': "E113°01.641'",
                'Month': 9,
                'Day': day_num,
                'Hour': hour_idx,
                'DayOfWeek': dt.weekday(),
                'DayOfYear': dt.timetuple().tm_yday,
                'Sea_Level_m': val
            })

    df = pd.DataFrame(records)

    X = df[['Day', 'Hour', 'DayOfWeek', 'DayOfYear']]
    y = df['Sea_Level_m']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    df['ML_Predicted_Sea_Level_m'] = np.round(model.predict(X), 2)
    return df, mae, r2

df, mae_score, r2_score_val = load_official_dishidros_dataset()

# PANELS NAVIGATION & TEST SIMULATOR
st.sidebar.markdown("### ⚙️ Panel Kontrol Navigasi")

# Opsi Simulasi untuk Tes Alarm Langsung
sim_low_water = st.sidebar.checkbox("🧪 Simulasi Level Air < 0.2m (Tes Alarm HP)")

current_day = now.day if now.month == 9 else 9
current_hour = now.hour

current_data = df[(df['Day'] == current_day) & (df['Hour'] == current_hour)]

if not current_data.empty:
    realtime_level = current_data['Sea_Level_m'].values[0]
    ml_level = current_data['ML_Predicted_Sea_Level_m'].values[0]
else:
    realtime_level = df.loc[0, 'Sea_Level_m']
    ml_level = df.loc[0, 'ML_Predicted_Sea_Level_m']

# Jika tombol simulasi dicentang, paksa level menjadi 0.15m
if sim_low_water:
    realtime_level = 0.15

prev_hour = current_hour - 1 if current_hour > 0 else 23
prev_day = current_day if current_hour > 0 else (current_day - 1 if current_day > 1 else 30)
prev_data = df[(df['Day'] == prev_day) & (df['Hour'] == prev_hour)]

if not prev_data.empty:
    prev_level = prev_data['Sea_Level_m'].values[0]
    trend_str = "🔻 SURUT" if realtime_level < prev_level else ("🔺 PASANG" if realtime_level > prev_level else "➖ STABIL")
else:
    trend_str = "➖ STABIL"

# =======================================================
# ALARM SUARA (WEB AUDIO API): BERBUNYI DI HP & LAPTOP
# =======================================================
if realtime_level < 0.2:
    # 1. Banner visual merah berkedip
    st.markdown(f"""
    <div class="alarm-banner">
        🚨 PERINGATAN CRITICAL: LEVEL AIR SANGAT LOW ({realtime_level:.2f} m < 0.20 m)!
    </div>
    """, unsafe_allow_html=True)

    # 2. Modul Audio HTML5 & JS (Bisa berbunyi di Chrome Mobile / HP & Laptop)
    components.html("""
        <div style="text-align: center; margin-top: 5px;">
            <button id="playBtn" onclick="enableAudio()" style="background: #ef4444; color: white; border: none; padding: 12px 24px; font-size: 16px; font-weight: bold; border-radius: 8px; cursor: pointer;">
                🔊 KLIK DISINI JIKA SUARA ALARM BELUM BUNYI DI HP
            </button>
        </div>

        <script>
            var audioCtx = null;
            var intervalId = null;

            function triggerSound() {
                try {
                    if (!audioCtx) {
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    }
                    if (audioCtx.state === 'suspended') {
                        audioCtx.resume();
                    }

                    var osc = audioCtx.createOscillator();
                    var gain = audioCtx.createGain();

                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // Frekuensi Nada A5
                    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.4);

                    gain.gain.setValueAtTime(0.5, audioCtx.currentTime);

                    osc.connect(gain);
                    gain.connect(audioCtx.destination);

                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.5);
                } catch(e) {
                    console.log("Audio waiting user touch");
                }
            }

            function enableAudio() {
                triggerSound();
                if(!intervalId) {
                    intervalId = setInterval(triggerSound, 1000);
                }
                document.getElementById("playBtn").style.display = "none";
            }

            // Mencoba membunyikan otomatis saat dibuka
            window.onload = function() {
                triggerSound();
                intervalId = setInterval(triggerSound, 1000);
            };
            
            // Trigger tambahan saat layar HP disentuh
            document.addEventListener('touchstart', function() {
                enableAudio();
            }, { once: true });
        </script>
    """, height=70)

# HEADER DASHBOARD
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>🌊 Smart Sea Water Level Monitoring & ML Analytics</h1>
            <p>📍 <strong>Stasiun Monitoring Intake Area PLTGU Grati</strong> — S7°38.659' E113°01.641'</p>
        </div>
        <div style="text-align: right;">
            <span class="badge-success">🟢 REALTIME ACTIVE</span><br>
            <small style="color: #94a3b8; font-size: 0.95rem;">Sync: {now.strftime('%H:%M:%S WIB')} (Auto 60s)</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# METRICS
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Waktu Monitoring</div>
        <div class="metric-value">{now.strftime('%H:%M')} <span style="font-size:1.1rem; color:#94a3b8;">WIB</span></div>
        <div class="metric-sub">{current_day} September 2026</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    color_style = "#ef4444" if realtime_level < 0.2 else "#38bdf8"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sea Level Realtime</div>
        <div class="metric-value" style="color: {color_style};">{realtime_level:.2f} <span style="font-size:1.1rem;">m</span></div>
        <div class="metric-sub">Kondisi: <strong>{trend_str}</strong></div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    delta_val = round(ml_level - realtime_level, 2)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Prediksi ML Model</div>
        <div class="metric-value" style="color: #f43f5e;">{ml_level:.2f} <span style="font-size:1.1rem;">m</span></div>
        <div class="metric-sub">Deviasi: <strong>{delta_val:+.2f} m</strong></div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Akurasi AI Model</div>
        <div class="metric-value" style="color: #10b981;">{r2_score_val*100:.1f}%</div>
        <div class="metric-sub">MAE Error: <strong>{mae_score:.3f} m</strong></div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Koordinat GPS</div>
        <div class="metric-value" style="color: #a855f7; font-size:1.4rem; margin-top:8px;">S7°38.659'</div>
        <div class="metric-sub">E113°01.641'</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# SIDEBAR OPTIONS
view_mode = st.sidebar.radio("Mode Tampilan Grafik:", options=["Mode Harian (24 Jam)", "Mode Per Jam (Detail 6 Jam)"])
selected_date = st.sidebar.date_input("Pilih Tanggal September 2026:", value=datetime.date(2026, 9, current_day), min_value=datetime.date(2026, 9, 1), max_value=datetime.date(2026, 9, 30))

st.sidebar.divider()
st.sidebar.info("""
**Stasiun:** Intake Area PLTGU Grati  
**Latitude:** S7°38.659' (-7.644317)  
**Longitude:** E113°01.641' (113.027350)  
**Zona Waktu:** GMT +07.00 (WIB)
""")

# GRAFIK & PETA STREAMLIT NATIVE (ANTI BLANK)
df_daily = df[(df['Day'] == selected_date.day)]
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📈 Hydro-Dynamic Curve ({selected_date.strftime('%d September 2026')})")
    df_plot = df_daily if view_mode == "Mode Harian (24 Jam)" else df_daily[(df_daily['Hour'] >= current_hour) & (df_daily['Hour'] <= current_hour + 6)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['Timestamp'], y=df_plot['Sea_Level_m'], mode='lines+markers', name='BMKG Hydro Baseline', line=dict(color='#0284c7', width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=df_plot['Timestamp'], y=df_plot['ML_Predicted_Sea_Level_m'], mode='lines+markers', name='AI ML Prediction', line=dict(color='#f43f5e', width=2.5, dash='dash'), marker=dict(size=7, symbol='x')))
    fig.add_trace(go.Scatter(x=[df_plot['Timestamp'].min(), df_plot['Timestamp'].max()], y=[0.2, 0.2], mode='lines', name='Threshold Critical (0.2m)', line=dict(color='#ef4444', width=2, dash='dot')))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15, 23, 42, 0.5)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor='#334155', showgrid=True),
        yaxis=dict(title='Tinggi Air Laut (Meter)', gridcolor='#334155', showgrid=True)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🗺️ Geospatial Intake Sensor")
    lat_dec, lon_dec = -7.644317, 113.027350
    
    # Menampilkan Peta Native Streamlit
    map_data = pd.DataFrame({'lat': [lat_dec], 'lon': [lon_dec]})
    st.map(map_data, zoom=14)

# DATAGRID
st.divider()
st.subheader("📊 Datagrid Telemetri & Export Laporan")
st.dataframe(df_plot[['Timestamp', 'Latitude', 'Longitude', 'Sea_Level_m', 'ML_Predicted_Sea_Level_m']], use_container_width=True)
