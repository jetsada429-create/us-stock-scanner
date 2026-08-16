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

# ================= ส่วนตั้งค่าแอป =================
# เพิ่มข้อความสถานะใหม่
STATUS_LABELS = {
    "UPTREND": {"text": "🚀 ขาขึ้นแข็งแกร่ง", "color": "badge-trend-bull"},
    "PULLBACK": {"text": "⏳ ย่อพักฐานขาขึ้น", "color": "badge-trend-pull"},
    "BUY_SUPPORT": {"text": "🎯 ช้อนแนวรับ", "color": "badge-trend-bull"},
    "SIDEWAYS": {"text": "〰️ ไซด์เวย์/สะสมแรง", "color": "badge-trend-side"},
    "DOWNTREND": {"text": "📉 ลงแรง/ขาลง", "color": "badge-trend-bear"},
}

# (CSS เดิมคงไว้ แต่เพิ่ม class ใหม่)
st.markdown(
    """
    <style>
    .badge-trend-bull { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
    .badge-trend-bear { background: #4C0519; color: #FDA4AF; border: 1px solid #9F1239; }
    .badge-trend-pull { background: #451A03; color: #FCD34D; border: 1px solid #78350F; }
    .badge-trend-side { background: #1E293B; color: #94A3B8; border: 1px solid #475569; }
    </style>
    """, unsafe_allow_html=True
)

@st.cache_data(ttl=14400)
def check_ma_snr_combo(ticker, info_mode=False):
    try:
        stock = yf.Ticker(ticker, session=get_yfinance_session())
        df = stock.history(period='2y', interval='1d')
        if len(df) < 50: return None, None
        df.columns = [col.lower() for col in df.columns]
        
        latest_close = df['close'].iloc[-1]
        fast_ma = df['close'].rolling(window=20).mean()
        slow_ma = df['close'].rolling(window=50).mean()
        
        # คำนวณ % ขาขึ้น (Bullish Score)
        bull_score = 0
        if latest_close >= fast_ma.iloc[-1]: bull_score += 30
        if latest_close >= slow_ma.iloc[-1]: bull_score += 20
        bullish_pct = min(100, max(0, bull_score))

        # คำนวณระยะย่อจาก High ล่าสุด 10 วัน
        recent_high_10d = df['high'].tail(10).max()
        drop_10d_pct = ((latest_close - recent_high_10d) / recent_high_10d) * 100
        
        # คำนวณระยะเด้งจาก Low 10 วัน
        recent_low_10d = df['low'].tail(10).min()
        bounce_10d_pct = ((latest_close - recent_low_10d) / recent_low_10d) * 100

        # --- ตรรกะใหม่: แยกสภาวะหุ้น ---
        # 1. ขาขึ้นแข็งแกร่ง
        if latest_close > slow_ma.iloc[-1] and drop_10d_pct > -10:
            trend_key = "UPTREND"
        # 2. ย่อพักฐานในขาขึ้น
        elif latest_close > slow_ma.iloc[-1] and drop_10d_pct <= -10:
            trend_key = "PULLBACK"
        # 3. ช้อนแนวรับ (เด้งจากโลว์)
        elif bounce_10d_pct >= 2.0 and latest_close < fast_ma.iloc[-1]:
            trend_key = "BUY_SUPPORT"
        # 4. ลงแรง
        elif drop_10d_pct < -20:
            trend_key = "DOWNTREND"
        # 5. สะสมตัว
        else:
            trend_key = "SIDEWAYS"

        status_info = STATUS_LABELS[trend_key]
        
        # จัดเตรียมข้อมูลส่งออกไปโชว์
        res_data = {
            'Ticker': ticker,
            'Price ($)': round(latest_close, 2),
            'bullish_pct': bullish_pct,
            'status_text': status_info['text'],
            'status_color': status_info['color'],
            # ... (ข้อมูลแนวรับ แนวต้าน) ...
        }
        return res_data, df
    except: return None, None
