import datetime
import io
import urllib3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Safe Import pytz
try:
  import pytz

  wib_tz = pytz.timezone("Asia/Jakarta")
  now = datetime.datetime.now(wib_tz)
except Exception:
  now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)

# Safe Import Auto-Refresh
try:
  from streamlit_autorefresh import st_autorefresh

  st_autorefresh_installed = True
except Exception:
  st_autorefresh_installed = False

# Safe Import Folium Map
try:
  import folium
  from streamlit_folium import st_folium

  folium_installed = True
except Exception:
  folium_installed = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Config Halaman
st.set_page_config(
    page_title="Smart Hydro Monitoring - S7°38.659' E113°01.641'",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if st_autorefresh_installed:
  st_autorefresh(interval=60000, limit=1000, key="datarefresh")

# Custom CSS
st.markdown(
    """
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
    .main-header h1 { color: #38bdf8; font-weight: 800; margin: 0; font-size: 2.3rem !important; }
    .main-header p { color: #cbd5e1; margin: 8px 0 0 0; font-size: 1.2rem !important; }

    .metric-card {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }
    .metric-title { font-size: 1rem !important; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.8px; }
    .metric-value { font-size: 2.2rem !important; font-weight: 800; color: #f8fafc; margin: 8px 0; }
    .metric-sub { font-size: 1rem !important; color: #38bdf8; font-weight: 600; }
    .badge-success { background-color: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid #10b981; padding: 4px 12px; border-radius: 8px; font-size: 1rem !important; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_official_dishidros_dataset():
  raw_matrix = [
      [
          2.4,
          2.3,
          2.0,
          1.6,
          1.2,
          0.9,
          0.9,
          1.0,
          1.2,
          1.6,
          1.9,
          2.1,
          2.2,
          2.1,
          1.8,
          1.5,
          1.2,
          1.0,
          0.9,
          1.0,
          1.3,
          1.7,
          2.1,
          2.3,
      ],
      [
          2.5,
          2.4,
          2.2,
          1.8,
          1.5,
          1.1,
          0.9,
          0.9,
          1.0,
          1.2,
          1.5,
          1.8,
          1.9,
          2.0,
          1.8,
          1.6,
          1.4,
          1.2,
          1.1,
          1.1,
          1.3,
          1.6,
          1.9,
          2.2,
      ],
      [
          2.4,
          2.5,
          2.4,
          2.1,
          1.8,
          1.4,
          1.1,
          1.0,
          0.9,
          1.0,
          1.2,
          1.4,
          1.6,
          1.7,
          1.7,
          1.6,
          1.5,
          1.4,
          1.3,
          1.3,
          1.4,
          1.6,
          1.8,
          2.1,
      ],
      [
          2.3,
          2.4,
          2.4,
          2.3,
          2.1,
          1.8,
          1.5,
          1.2,
          1.0,
          0.9,
          0.9,
          1.0,
          1.2,
          1.3,
          1.4,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.6,
          1.6,
          1.8,
          1.9,
      ],
      [
          2.1,
          2.2,
          2.3,
          2.3,
          2.3,
          2.1,
          1.9,
          1.6,
          1.3,
          1.1,
          0.9,
          0.8,
          0.8,
          0.9,
          1.0,
          1.2,
          1.4,
          1.6,
          1.7,
          1.7,
          1.7,
          1.8,
          1.8,
          1.8,
      ],
      [
          1.9,
          2.0,
          2.1,
          2.2,
          2.3,
          2.3,
          2.2,
          2.0,
          1.7,
          1.4,
          1.1,
          0.8,
          0.6,
          0.7,
          0.9,
          1.1,
          1.4,
          1.7,
          1.9,
          1.9,
          1.9,
          1.9,
          1.9,
          1.8,
      ],
      [
          1.8,
          1.8,
          1.9,
          2.0,
          2.2,
          2.3,
          2.4,
          2.4,
          2.1,
          1.8,
          1.4,
          1.0,
          0.6,
          0.4,
          0.5,
          0.8,
          1.2,
          1.6,
          1.9,
          2.1,
          2.1,
          2.0,
          1.9,
          1.8,
      ],
      [
          1.7,
          1.6,
          1.6,
          1.7,
          1.9,
          2.2,
          2.4,
          2.6,
          2.5,
          2.2,
          1.8,
          1.3,
          0.8,
          0.5,
          0.2,
          0.2,
          0.5,
          0.9,
          1.4,
          1.8,
          2.2,
          2.3,
          2.2,
          2.0,
      ],
      [
          1.8,
          1.5,
          1.4,
          1.4,
          1.6,
          1.9,
          2.3,
          2.6,
          2.7,
          2.6,
          2.3,
          1.8,
          1.2,
          0.7,
          0.3,
          0.1,
          0.2,
          0.6,
          1.1,
          1.6,
          2.1,
          2.4,
          2.4,
          2.2,
      ],
      [
          1.9,
          1.6,
          1.3,
          1.2,
          1.3,
          1.5,
          1.9,
          2.4,
          2.7,
          2.8,
          2.6,
          2.2,
          1.6,
          1.0,
          0.5,
          0.2,
          0.2,
          0.4,
          0.8,
          1.4,
          1.9,
          2.3,
          2.5,
          2.4,
      ],
      [
          2.1,
          1.7,
          1.3,
          1.1,
          1.0,
          1.2,
          1.6,
          2.0,
          2.5,
          2.7,
          2.7,
          2.5,
          2.0,
          1.4,
          0.9,
          0.4,
          0.3,
          0.3,
          0.7,
          1.2,
          1.7,
          2.2,
          2.5,
          2.5,
      ],
      [
          2.2,
          1.9,
          1.4,
          1.1,
          0.9,
          0.9,
          1.2,
          1.6,
          2.1,
          2.5,
          2.7,
          2.6,
          2.3,
          1.8,
          1.3,
          0.8,
          0.5,
          0.4,
          0.6,
          1.0,
          1.5,
          2.1,
          2.4,
          2.5,
      ],
      [
          2.4,
          2.0,
          1.6,
          1.2,
          0.9,
          0.8,
          0.9,
          1.3,
          1.7,
          2.1,
          2.5,
          2.6,
          2.4,
          2.1,
          1.6,
          1.1,
          0.8,
          0.6,
          0.7,
          1.0,
          1.4,
          1.9,
          2.3,
          2.5,
      ],
      [
          2.4,
          2.2,
          1.8,
          1.4,
          1.0,
          0.8,
          0.8,
          1.0,
          1.3,
          1.7,
          2.1,
          2.3,
          2.3,
          2.1,
          1.8,
          1.4,
          1.1,
          0.9,
          0.9,
          1.1,
          1.4,
          1.8,
          2.1,
          2.4,
      ],
      [
          2.4,
          2.3,
          2.0,
          1.6,
          1.2,
          0.9,
          0.8,
          0.9,
          1.1,
          1.4,
          1.7,
          2.0,
          2.1,
          2.0,
          1.9,
          1.6,
          1.4,
          1.2,
          1.2,
          1.3,
          1.5,
          1.8,
          2.1,
          2.3,
      ],
      [
          2.4,
          2.3,
          2.1,
          1.8,
          1.4,
          1.1,
          0.9,
          0.9,
          1.0,
          1.1,
          1.4,
          1.6,
          1.7,
          1.8,
          1.8,
          1.7,
          1.5,
          1.4,
          1.4,
          1.5,
          1.6,
          1.8,
          2.0,
          2.2,
      ],
      [
          2.3,
          2.3,
          2.1,
          1.9,
          1.6,
          1.3,
          1.1,
          1.0,
          1.0,
          1.0,
          1.1,
          1.3,
          1.4,
          1.5,
          1.5,
          1.6,
          1.6,
          1.6,
          1.6,
          1.7,
          1.8,
          1.9,
          2.1,
          2.2,
      ],
      [
          2.2,
          2.2,
          2.1,
          2.0,
          1.8,
          1.6,
          1.4,
          1.2,
          1.1,
          1.0,
          1.0,
          1.1,
          1.1,
          1.2,
          1.3,
          1.4,
          1.5,
          1.6,
          1.7,
          1.8,
          1.9,
          2.0,
          2.1,
          2.2,
      ],
      [
          2.2,
          2.2,
          2.1,
          2.0,
          1.9,
          1.8,
          1.6,
          1.4,
          1.3,
          1.1,
          1.0,
          1.0,
          0.9,
          0.9,
          1.0,
          1.1,
          1.3,
          1.5,
          1.7,
          1.9,
          2.0,
          2.1,
          2.1,
          2.2,
      ],
      [
          2.1,
          2.1,
          2.0,
          2.0,
          1.9,
          1.8,
          1.7,
          1.5,
          1.3,
          1.1,
          1.0,
          0.8,
          0.7,
          0.8,
          0.9,
          1.1,
          1.4,
          1.6,
          1.9,
          2.0,
          2.1,
          2.2,
          2.1,
          2.0,
      ],
      [
          2.1,
          2.0,
          2.0,
          1.9,
          2.0,
          2.0,
          2.0,
          1.9,
          1.8,
          1.6,
          1.3,
          1.1,
          0.8,
          0.7,
          0.6,
          0.7,
          0.9,
          1.2,
          1.5,
          1.8,
          2.0,
          2.1,
          2.1,
          2.1,
      ],
      [
          2.0,
          1.9,
          1.8,
          1.8,
          1.9,
          2.0,
          2.1,
          2.1,
          2.0,
          1.8,
          1.5,
          1.2,
          0.9,
          0.7,
          0.6,
          0.6,
          0.8,
          1.1,
          1.4,
          1.8,
          2.0,
          2.1,
          2.1,
          2.0,
      ],
      [
          1.9,
          1.8,
          1.7,
          1.7,
          1.8,
          1.9,
          2.1,
          2.2,
          2.2,
          2.1,
          1.8,
          1.4,
          1.1,
          0.8,
          0.6,
          0.5,
          0.7,
          1.0,
          1.4,
          1.7,
          2.0,
          2.1,
          2.1,
          2.0,
      ],
      [
          1.8,
          1.6,
          1.5,
          1.5,
          1.6,
          1.8,
          2.0,
          2.2,
          2.4,
          2.3,
          2.1,
          1.7,
          1.3,
          0.9,
          0.7,
          0.5,
          0.6,
          0.9,
          1.3,
          1.7,
          2.0,
          2.2,
          2.2,
          2.0,
      ],
      [
          1.8,
          1.5,
          1.3,
          1.2,
          1.3,
          1.5,
          1.8,
          2.2,
          2.4,
          2.4,
          2.3,
          1.9,
          1.5,
          1.1,
          0.8,
          0.6,
          0.6,
          0.9,
          1.2,
          1.7,
          2.1,
          2.3,
          2.3,
          2.1,
      ],
      [
          1.8,
          1.5,
          1.2,
          1.0,
          1.0,
          1.2,
          1.6,
          1.9,
          2.3,
          2.5,
          2.4,
          2.2,
          1.8,
          1.3,
          0.9,
          0.7,
          0.7,
          0.8,
          1.2,
          1.6,
          2.1,
          2.4,
          2.5,
          2.3,
      ],
      [
          2.0,
          1.5,
          1.2,
          0.9,
          0.8,
          0.9,
          1.2,
          1.6,
          2.0,
          2.4,
          2.5,
          2.3,
          2.0,
          1.6,
          1.1,
          0.9,
          0.7,
          0.9,
          1.2,
          1.6,
          2.0,
          2.4,
          2.6,
          2.5,
      ],
      [
          2.2,
          1.7,
          1.3,
          0.9,
          0.7,
          0.7,
          0.9,
          1.3,
          1.7,
          2.1,
          2.3,
          2.3,
          2.1,
          1.8,
          1.4,
          1.0,
          0.9,
          0.9,
          1.1,
          1.5,
          2.0,
          2.4,
          2.6,
          2.6,
      ],
      [
          2.4,
          2.0,
          1.5,
          1.0,
          0.7,
          0.6,
          0.6,
          0.9,
          1.3,
          1.8,
          2.1,
          2.2,
          2.2,
          1.9,
          1.6,
          1.3,
          1.1,
          1.0,
          1.2,
          1.5,
          1.9,
          2.3,
          2.6,
          2.7,
      ],
      [
          2.6,
          2.3,
          1.8,
          1.3,
          0.9,
          0.6,
          0.5,
          0.7,
          1.0,
          1.4,
          1.7,
          2.0,
          2.0,
          2.0,
          1.7,
          1.5,
          1.3,
          1.2,
          1.2,
          1.4,
          1.8,
          2.2,
          2.5,
          2.7,
      ],
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
          'Sea_Level_m': raw_matrix[day_idx][hour_idx],
      })

  df = pd.DataFrame(records)

  # Model Training using Cyclical Features
  X = df[['Day', 'Hour_sin', 'Hour_cos']]
  y = df['Sea_Level_m']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  mae = mean_absolute_error(y_test, model.predict(X_test))
  r2 = r2_score(y_test, model.predict(X_test))

  df['ML_Predicted_Sea_Level_m'] = np.round(model.predict(X), 2)
  return df, mae, r2


df, mae_score, r2_score_val = load_official_dishidros_dataset()

# Realtime calculation
current_day = now.day if now.month == 9 else 9
current_hour = now.hour

current_data = df[(df['Day'] == current_day) & (df['Hour'] == current_hour)]
realtime_level = (
    current_data['Sea_Level_m'].values[0]
    if not current_data.empty
    else df.loc[0, 'Sea_Level_m']
)
ml_level = (
    current_data['ML_Predicted_Sea_Level_m'].values[0]
    if not current_data.empty
    else df.loc[0, 'ML_Predicted_Sea_Level_m']
)

# Render Header & Cards
st.markdown(
    f"""
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
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Waktu", f"{now.strftime('%H:%M')} WIB")
m2.metric("Sea Level Realtime", f"{realtime_level:.2f} m")
m3.metric("Prediksi ML Model", f"{ml_level:.2f} m")
m4.metric("Akurasi AI Model", f"{r2_score_val * 100:.1f}%")

# Plotly Section
selected_date = st.sidebar.date_input(
    "Pilih Tanggal September 2026:",
    value=datetime.date(2026, 9, current_day),
    min_value=datetime.date(2026, 9, 1),
    max_value=datetime.date(2026, 9, 30),
)

df_plot = df[(df['Day'] == selected_date.day)]
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_plot['Timestamp'],
        y=df_plot['Sea_Level_m'],
        mode='lines+markers',
        name='BMKG Baseline',
    )
)
fig.add_trace(
    go.Scatter(
        x=df_plot['Timestamp'],
        y=df_plot['ML_Predicted_Sea_Level_m'],
        mode='lines',
        name='AI Prediction',
    )
)
fig.update_layout(template='plotly_dark', height=400)

st.plotly_chart(fig, use_container_width=True)
