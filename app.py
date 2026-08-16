import concurrent.futures
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# ================= โครงสร้างพื้นฐานของแอป =================
st.set_page_config(page_title="US Stock Scanner PRO", layout="wide")

# ป้องกัน Global State Error
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.is_scanning = False
    st.session_state.latest_results = None
    st.session_state.watchlist = []

# ================= ฟังก์ชันหลัก (รวมการแก้ไขความแข็งแรง) =================
@st.cache_resource
def get_yfinance_session():
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    adapter = HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)
    return session

@st.cache_data(ttl=3600)
def get_us_stock_tickers():
    """ดึง tickers โดยมี error handling กันแอปขาว"""
    try:
        # ใช้ URL ที่เข้าถึงง่ายขึ้นแทน FTP หากมีปัญหา
        url = 'https://dumbstockapi.com/stock?exchange=NASDAQ,NYSE'
        response = requests.get(url, timeout=10)
        data = response.json()
        return [stock['ticker'] for stock in data]
    except:
        # หากดึงไม่ได้ ให้ใช้ตัวอย่างเริ่มต้นเพื่อไม่ให้แอปล่ม
        return ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT', 'PLTR', 'RKLB', 'IREN', 'AAOI']

def calculate_swing_snr(df, latest_close):
    # ปรับปรุงสูตร: ใช้ window 120 วัน และ Pivot Low ล่าสุด ไม่ย้อนไปไกลเกิน
    n = len(df)
    df_wave = df.iloc[-120:]
    wave_high = float(np.max(df_wave['high']))
    wave_low = float(np.min(df_wave['low']))
    wave_range = max(1e-4, wave_high - wave_low)

    # Fibonacci ระดับมาตรฐาน
    fib_382 = wave_high - 0.382 * wave_range
    fib_500 = wave_high - 0.500 * wave_range
    fib_618 = wave_high - 0.618 * wave_range

    s1 = float(np.min(df_wave['low'].tail(20)))
    s1 = s1 if s1 < latest_close * 0.99 else latest_close * 0.95
    s2 = fib_618 if fib_618 < s1 * 0.95 else s1 * 0.90
    s3 = wave_low if wave_low < s2 * 0.90 else s2 * 0.85
    
    r1 = float(np.max(df_wave['high'].tail(20)))
    r1 = r1 if r1 > latest_close * 1.02 else latest_close * 1.05
    r2 = fib_382 if fib_382 > r1 else r1 * 1.10
    r3 = fib_500 * 1.1 if fib_500 > r1 else r1 * 1.15
    r4 = wave_high if wave_high > r1 else r1 * 1.25

    return round(s1, 2), round(s2, 2), round(s3, 2), round(r1, 2), round(r2, 2), round(r3, 2), round(r4, 2)

# ================= หน้าจอหลัก =================
try:
    st.markdown("<h1 style='text-align: center;'>US Stock Scanner PRO</h1>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔍 วิเคราะห์", "🚀 สแกน", "⭐ Watchlist"])

    with tab1:
        ticker = st.text_input("ระบุ Ticker:", "AAOI").upper()
        if st.button("วิเคราะห์หุ้น"):
            # เรียกฟังก์ชันใน try-except เพื่อดู error หากมี
            res, df = check_ma_snr_combo(ticker, info_mode=True)
            if res:
                st.success(f"พบข้อมูล {ticker}")
                st.write(res)
            else:
                st.error("ไม่พบข้อมูลหุ้นตัวนี้")

    with tab2:
        st.write("กดปุ่มสแกนเพื่อเริ่มวิเคราะห์ตลาด")
        if st.button("🚀 สแกนทั้งตลาด"):
            st.session_state.is_scanning = True
            st.write("กำลังสแกน... (อาจใช้เวลาสักครู่)")
            # เพิ่ม Logic สแกนของคุณที่นี่
            st.session_state.is_scanning = False

except Exception as e:
    st.error(f"ระบบขัดข้อง: {e}")
    st.write("กรุณาแจ้งข้อความนี้เพื่อทำการแก้ไขครับ")
