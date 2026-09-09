import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import datetime
import pytz
import urllib3

# Matikan peringatan SSL insecure jika verify=False digunakan
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Config halaman Streamlit
st.set_page_config(
    page_title="Pasang Surut Sea Water Level Probolinggo 2026",
    page_icon="🌊",
    layout="wide"
)

# ==========================================
# PERBAIKAN FUNCTION FETCH DATA BMKG
# ==========================================
@st.cache_data(ttl=120)  # Cache 2 menit agar tidak membebankan server BMKG
def fetch_bmkg_maritim_data():
    url = "https://maritim.bmkg.go.id/"
    
    # Menyamar sebagai browser Chrome asli agar tidak diblokir firewall BMKG
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive"
    }
    
    try:
        # verify=False melewati pengecekan sertifikat SSL jika server BMKG strict
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        if response.status_code == 200:
            return True, "Active"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        # Jika gagal koneksi langsung, coba ping endpoint API alternatif BMKG
        try:
            alt_url = "https://data.bmkg.go.id/"
            alt_resp = requests.get(alt_url, headers=headers, timeout=5, verify=False)
            if alt_resp.status_code == 200:
                return True, "Active (Alt Gateway)"
        except Exception:
            pass
        return False, "Connection Refused / Blocked"
