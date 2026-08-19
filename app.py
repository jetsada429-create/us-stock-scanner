import concurrent.futures
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= 1. ตั้งค่าแอปและตัวแปรหลัก =================
st.set_page_config(
    page_title="US Stock & Forex Scanner PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True
}

UI_LANG_MAP = {
    'search_ticker_title': "US Stock & Forex Scanner PRO (by.Jetsada)",
    'search_ticker_subtitle': "ระบบสแกนเทคนิคอล • วิเคราะห์กระแสเงิน Nasdaq & S&P500 • AI Pattern • Forex & ทองคำ",
    'search_ticker_label': "พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, RKLB, AAOI, RXT, CRWV, BZAI, TSM):",
    'search_forex_label': "พิมพ์คู่เงินหรือสินทรัพย์ (เช่น XAUUSD, EURUSD, GBPUSD, USDJPY, BTCUSD, USOIL):",
    'btn_analyze_single': "🔎 วิเคราะห์ทันที",
    'btn_scan_market': "🚀 เริ่มสแกนตลาดหุ้น",
    'btn_scan_forex': "🚀 สแกนตลาด Forex & ทองคำ",
    'status_preparing_tickers': "⏳ กำลังดึงรายชื่อหุ้นผู้นำตลาด...",
    'status_scanning': "⏳ สแกนไปแล้ว {count}/{total} ตัว (พบหุ้นทรงสวย {found} ตัว)...",
    'status_analyzing_single': "⏳ กำลังดึงข้อมูลและวิเคราะห์ {ticker}...",
    'expander_business_summary': "📖 สรุปธุรกิจ & โครงสร้างผู้ถือหุ้น (แปลไทยอัตโนมัติ)",
    'chart_title_single': "📈 กราฟเทคนิค 3 แนวรับ และ 4 ระดับแนวต้าน",
    'analysis_title': "📊 ข้อมูลแนวรับ - แนวต้าน & ตัวชี้วัดสำคัญ",
    'tab_market_flow': "🏛️ ทิศทางตลาด Nasdaq & S&P 500",
    'tab_search_ticker': "🔍 ค้นหา & วิเคราะห์หุ้นรายตัว",
    'tab_scan_market': "🚀 สแกนคัดหุ้นทรงสวย",
    'tab_forex': "💱 วิเคราะห์ Forex & ทองคำ (XAUUSD)",
    'tab_macro_news': "📰 ข่าวเด่นเศรษฐกิจ & ปัจจัยตลาดหุ้น",
}

SECTOR_MAP_TH = {
    'Technology': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์',
    'Information Technology': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์',
    'Healthcare': '🏥 สุขภาพ / การแพทย์ & ยา',
    'Health Care': '🏥 สุขภาพ / การแพทย์ & ยา',
    'Financial Services': '🏦 การเงิน / ธนาคาร & ประกันภัย',
    'Financials': '🏦 การเงิน / ธนาคาร & ประกันภัย',
    'Industrials': '🏭 อุตสาหกรรม / อวกาศ & ขนส่ง',
    'Consumer Cyclical': '🛍️ สินค้าฟุ่มเฟือย / ค้าปลีก & ยานยนต์',
    'Consumer Discretionary': '🛍️ สินค้าฟุ่มเฟือย / ค้าปลีก & ยานยนต์',
    'Consumer Defensive': '🛒 สินค้าอุปโภคบริโภคจำเป็น',
    'Consumer Staples': '🛒 สินค้าอุปโภคบริโภคจำเป็น',
    'Energy': '⚡ พลังงาน / น้ำมัน & ก๊าซ',
    'Real Estate': '🏢 อสังหาริมทรัพย์ / กองรีท (REITs)',
    'Basic Materials': '🧪 วัตถุดิบพื้นฐาน / เคมีภัณฑ์ & เหมืองแร่',
    'Materials': '🧪 วัตถุดิบพื้นฐาน / เคมีภัณฑ์ & เหมืองแร่',
    'Communication Services': '📡 สื่อสาร / โทรคมนาคม & บันเทิง',
    'Utilities': '💡 สาธารณูปโภค / ไฟฟ้า & ประปา'
}

FOREX_DIRECTORY = [
    {'ticker': 'XAUUSD', 'name': 'Gold Spot / US Dollar (ทองคำ)', 'type': 'Commodity', 'exchange': 'Precious Metals'},
    {'ticker': 'EURUSD', 'name': 'Euro / US Dollar', 'type': 'Major Forex', 'exchange': 'Forex Market'},
    {'ticker': 'GBPUSD', 'name': 'British Pound / US Dollar', 'type': 'Major Forex', 'exchange': 'Forex Market'},
    {'ticker': 'USDJPY', 'name': 'US Dollar / Japanese Yen', 'type': 'Major Forex', 'exchange': 'Forex Market'},
    {'ticker': 'AUDUSD', 'name': 'Australian Dollar / US Dollar', 'type': 'Major Forex', 'exchange': 'Forex Market'},
    {'ticker': 'USDCAD', 'name': 'US Dollar / Canadian Dollar', 'type': 'Major Forex', 'exchange': 'Forex Market'},
    {'ticker': 'USDCHF', 'name': 'US Dollar / Swiss Franc', 'type': 'Major Forex', 'exchange': 'Forex Market'},
    {'ticker': 'NZDUSD', 'name': 'New Zealand Dollar / US Dollar', 'type': 'Major Forex', 'exchange': 'Forex Market'},
    {'ticker': 'EURJPY', 'name': 'Euro / Japanese Yen', 'type': 'Cross Forex', 'exchange': 'Forex Market'},
    {'ticker': 'GBPJPY', 'name': 'British Pound / Japanese Yen', 'type': 'Cross Forex', 'exchange': 'Forex Market'},
    {'ticker': 'BTCUSD', 'name': 'Bitcoin / US Dollar (บิตคอยน์)', 'type': 'Crypto', 'exchange': 'Crypto Market'},
    {'ticker': 'ETHUSD', 'name': 'Ethereum / US Dollar (อีเธอเรียม)', 'type': 'Crypto', 'exchange': 'Crypto Market'},
    {'ticker': 'USOIL', 'name': 'Crude Oil WTI (น้ำมันดิบ)', 'type': 'Commodity', 'exchange': 'Energy Commodity'},
    {'ticker': 'XAGUSD', 'name': 'Silver Spot / US Dollar (โลหะเงิน)', 'type': 'Commodity', 'exchange': 'Precious Metals'}
]

# ================= 2. ฟังก์ชันตัวช่วยทั้งหมด (Top-Level Helpers) =================
def get_time_elapsed_thai(last_dt):
    if not last_dt:
        return ""
    try:
        diff = datetime.now() - last_dt
        secs = int(diff.total_seconds())
        if secs < 60:
            return f" (เพิ่งสแกนเมื่อ {secs} วิที่แล้ว)"
        elif secs < 3600:
            return f" (สแกนไปแล้ว {secs // 60} นาทีที่แล้ว)"
        else:
            return f" (สแกนไปแล้ว {secs // 3600} ชม. ก่อน)"
    except Exception:
        return ""

def translate_text_to_thai(text):
    if not text or text == 'N/A' or not str(text).strip():
        return ''
    text_sample = str(text)[:350]
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "en", "tl": "th", "dt": "t", "q": text_sample}
        res = requests.get(url, params=params, timeout=2.5)
        if res.status_code == 200:
            return "".join([item[0] for item in res.json()[0] if item[0]])
    except Exception:
        pass
    return str(text_sample)

def resolve_financial_symbol(ticker_str):
    raw = str(ticker_str).strip().upper()
    mapping = {
        'XAUUSD': 'GC=F', 'GOLD': 'GC=F', 'XAU': 'GC=F',
        'XAGUSD': 'SI=F', 'SILVER': 'SI=F',
        'USOIL': 'CL=F', 'OIL': 'CL=F', 'WTI': 'CL=F',
        'BTCUSD': 'BTC-USD', 'BTC': 'BTC-USD',
        'ETHUSD': 'ETH-USD', 'ETH': 'ETH-USD',
        'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
        'AUDUSD': 'AUDUSD=X', 'USDCAD': 'USDCAD=X', 'USDCHF': 'USDCHF=X',
        'NZDUSD': 'NZDUSD=X', 'EURJPY': 'EURJPY=X', 'GBPJPY': 'GBPJPY=X'
    }
    if raw in mapping:
        return mapping[raw], raw
    if len(raw) == 6 and (raw.endswith('USD') or raw.startswith('USD') or raw.startswith('EUR') or raw.startswith('GBP')):
        return f"{raw}=X", raw
    return raw, raw

@st.cache_resource
def get_yfinance_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))
    return session

@st.cache_resource
def get_global_server_state():
    return {
        "is_scanning": False,
        "scan_start_time": None,
        "latest_results": None,
        "latest_df": None,
        "last_scanned_at": None,
        "last_scanned_dt": None,
        "forex_results": None,
        "forex_df": None,
        "forex_scanned_at": None
    }

server_state = get_global_server_state()

# ================= 3. Custom CSS ปรับแต่งสีสันให้คมชัดโดดเด่น =================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 1200px;
    }
    .main-title {
        font-size: 1.55rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 0.78rem !important;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        padding: 0.45rem 0.8rem !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        opacity: 0.92;
        transform: translateY(-1px);
    }
    
    .status-banner {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 0.8rem;
        line-height: 1.5;
        box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    }
    .status-banner-uptrend { background-color: #022c22 !important; border: 1.5px solid #10b981 !important; color: #a7f3d0 !important; }
    .status-banner-pullback { background-color: #451a03 !important; border: 1.5px solid #f59e0b !important; color: #fef08a !important; }
    .status-banner-support { background-color: #172554 !important; border: 1.5px solid #38bdf8 !important; color: #bae6fd !important; }
    .status-banner-sideways { background-color: #1e293b !important; border: 1.5px solid #94a3b8 !important; color: #f1f5f9 !important; }
    .status-banner-downtrend { background-color: #4c0519 !important; border: 1.5px solid #f43f5e !important; color: #fecdd3 !important; }
    .status-title-text { font-size: 1.0rem; font-weight: 800; margin-bottom: 4px; letter-spacing: -0.2px; }
    .status-desc-text { font-size: 0.84rem; opacity: 0.95; }

    .compact-board {
        background: #0B132B;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 0.5rem;
    }
    .price-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 8px;
    }
    .price-val-box { display: flex; align-items: baseline; gap: 6px; }
    .price-main { font-size: 1.45rem; font-weight: 900; color: #F8FAFC; }
    .price-badge-group { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
    .price-badge { font-size: 0.76rem; padding: 3px 8px; border-radius: 6px; font-weight: 700; white-space: nowrap; }
    .badge-rsi { background: #1E293B; color: #38BDF8; border: 1px solid #334155; }
    .badge-dist { background: #064E3B; color: #34D399; border: 1px solid #059669; }
    
    .badge-board-uptrend { background: #059669 !important; color: #FFFFFF !important; border: 1.5px solid #34D399 !important; font-weight: 800 !important; }
    .badge-board-pullback { background: #D97706 !important; color: #FFFFFF !important; border: 1.5px solid #FCD34D !important; font-weight: 800 !important; }
    .badge-board-support { background: #0284C7 !important; color: #FFFFFF !important; border: 1.5px solid #38BDF8 !important; font-weight: 800 !important; }
    .badge-board-sideways { background: #475569 !important; color: #FFFFFF !important; border: 1.5px solid #94A3B8 !important; font-weight: 800 !important; }
    .badge-board-downtrend { background: #E11D48 !important; color: #FFFFFF !important; border: 1.5px solid #FDA4AF !important; font-weight: 800 !important; }

    .badge-ai-box { background: #172554; color: #93C5FD; border: 1px solid #1E40AF; font-weight: 700; }
    .badge-market { background: #1e1b4b; color: #c7d2fe; border: 1px solid #4338ca; font-weight: 700; }

    .snr-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .snr-card { background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 6px 8px; }
    .snr-card-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; padding-bottom: 3px; border-bottom: 1px dashed #334155; }
    .snr-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.76rem; padding: 2px 0; }
    .snr-lbl { color: #94A3B8; font-size: 0.72rem; }
    .snr-num { font-weight: 700; font-size: 0.82rem; }
    
    .c-green { color: #22C55E !important; }
    .c-lightgreen { color: #4ADE80 !important; }
    .c-red { color: #EF4444 !important; }
    .c-orange { color: #F97316 !important; }
    .c-yellow { color: #FBBF24 !important; }
    .c-darkred { color: #F43F5E !important; }

    .strategy-card { background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
    .strat-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .strat-title { font-size: 0.82rem; font-weight: 700; color: #F8FAFC; }
    .strat-price { font-size: 0.95rem; font-weight: 800; color: #38BDF8; }
    .strat-body { display: flex; justify-content: space-between; align-items: center; font-size: 0.74rem; padding-top: 4px; border-top: 1px dashed #1E293B; flex-wrap: wrap; gap: 4px; }
    .strat-sub { color: #94A3B8; }
    .strat-val { color: #F8FAFC; font-weight: 600; }

    .company-header { font-size: 1.15rem; font-weight: 800; color: #38BDF8 !important; margin-bottom: 0rem; }
    .sector-badge { font-size: 0.75rem; font-weight: 600; color: #FCD34D; background: #451A03; border: 1px solid #78350F; padding: 3px 7px; border-radius: 5px; display: inline-block; margin-top: 3px; margin-bottom: 4px; }
    .chart-header-badge { font-size: 0.82rem; font-weight: 700; color: #F8FAFC; background-color: #1E293B; padding: 4px 7px; border-radius: 5px; margin-bottom: 3px; display: inline-block; }
    .fin-card { background: #0F172A !important; border: 1px solid #334155 !important; border-radius: 8px; padding: 10px 12px; margin-bottom: 0.4rem; color: #F8FAFC !important; }
    .biz-summary { font-size: 0.86rem !important; color: #F8FAFC !important; background-color: #0B132B !important; padding: 12px 14px !important; border-radius: 8px; border-left: 4px solid #3B82F6 !important; border: 1px solid #334155 !important; margin-top: 6px; margin-bottom: 0.5rem; line-height: 1.6; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .pattern-box { background-color: #172554 !important; color: #93C5FD !important; padding: 5px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 600; border: 1px solid #1E40AF !important; margin-top: 3px; margin-bottom: 4px; }
    
    .market-flow-card {
        background: #0B132B;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .flow-meter-container {
        background: #1E293B;
        border-radius: 8px;
        height: 22px;
        width: 100%;
        display: flex;
        overflow: hidden;
        margin: 6px 0 10px 0;
        border: 1px solid #334155;
    }
    .flow-buy-bar { background: linear-gradient(90deg, #10B981, #059669); height: 100%; text-align: center; color: #fff; font-size: 0.75rem; font-weight: bold; line-height: 22px; }
    .flow-sell-bar { background: linear-gradient(90deg, #E11D48, #BE123C); height: 100%; text-align: center; color: #fff; font-size: 0.75rem; font-weight: bold; line-height: 22px; }
    .flow-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 8px; }
    .flow-sub-card { background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 8px; text-align: center; }
    
    .danger-alert-box {
        background: #2A0814;
        border: 1px solid #E11D48;
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 10px;
        font-size: 0.78rem;
        line-height: 1.5;
        color: #FECDD3;
    }

    .news-card-link {
        background: #0B132B;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
        display: block;
        text-decoration: none;
        transition: all 0.2s ease-in-out;
    }
    .news-card-link:hover {
        border-color: #3B82F6;
        transform: translateX(3px);
        background: #0F172A;
    }
    .news-card-title { font-size: 0.88rem; font-weight: 700; color: #60A5FA !important; margin-bottom: 4px; line-height: 1.4; }
    .news-card-meta { font-size: 0.72rem; color: #94A3B8; }

    .desktop-only-space { height: 28px; display: block; }
    @media (max-width: 640px) {
        .desktop-only-space { display: none !important; }
        .main-title { font-size: 1.3rem !important; }
        .price-main { font-size: 1.3rem; }
        .flow-grid-3 { grid-template-columns: 1fr; }
        .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="main-title">{UI_LANG_MAP["search_ticker_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{UI_LANG_MAP["search_ticker_subtitle"]}</div>', unsafe_allow_html=True)

# ================= 5. ฟังก์ชันวิเคราะห์กระแสเงินและสร้างกราฟเม็ดเงิน (Nasdaq & S&P 500) =================
@st.cache_data(ttl=900, show_spinner=False)
def calculate_market_flow_advanced(index_symbol, index_name):
    try:
        stock = yf.Ticker(index_symbol, session=get_yfinance_session())
        df = stock.history(period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 25:
            return None, None
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        vol = df['Volume']
        
        price_range = (high - low).replace(0, 1e-4)
        mfm = ((close - low) - (high - close)) / price_range
        
        net_vol = mfm * vol
        df['cum_money_flow'] = (net_vol / 1_000_000).cumsum()
        
        df['cmf_20'] = (net_vol.rolling(20).sum()) / (vol.rolling(20).sum().replace(0, 1e-4))
        latest_cmf = round(float(df['cmf_20'].iloc[-1]), 2)
        
        buy_vol = vol * ((1.0 + mfm) / 2.0)
        sell_vol = vol * ((1.0 - mfm) / 2.0)
        d1_buy = float(buy_vol.iloc[-1])
        d1_sell = float(sell_vol.iloc[-1])
        d1_tot = max(1.0, d1_buy + d1_sell)
        d1_buy_pct = round((d1_buy / d1_tot) * 100, 1)
        d1_sell_pct = round(100.0 - d1_buy_pct, 1)
        d1_winner = "ฝั่งซื้อชนะ" if d1_buy_pct >= 50 else "ฝั่งขายคุม"
        
        w1_buy = float(buy_vol.tail(5).sum())
        w1_sell = float(sell_vol.tail(5).sum())
        w1_tot = max(1.0, w1_buy + w1_sell)
        w1_buy_pct = round((w1_buy / w1_tot) * 100, 1)
        w1_sell_pct = round(100.0 - w1_buy_pct, 1)
        w1_winner = f"ฝั่งซื้อสะสมนำ ({w1_buy_pct}%)" if w1_buy_pct >= 50 else f"ฝั่งขายสะสมนำ ({w1_sell_pct}%)"
        
        m1_buy = float(buy_vol.tail(21).sum())
        m1_sell = float(sell_vol.tail(21).sum())
        m1_tot = max(1.0, m1_buy + m1_sell)
        m1_buy_pct = round((m1_buy / m1_tot) * 100, 1)
        m1_sell_pct = round(100.0 - m1_buy_pct, 1)
        m1_winner = f"ยอดสะสมซื้อนำ ({m1_buy_pct}%)" if m1_buy_pct >= 50 else f"ยอดสะสมขายนำ ({m1_sell_pct}%)"
        
        latest_price = round(float(close.iloc[-1]), 2)
        prev_price = float(close.iloc[-2])
        chg_pct = round(((latest_price - prev_price) / prev_price) * 100, 2)
        
        ma20_val = round(float(close.rolling(20).mean().iloc[-1]), 2)
        support_5d = round(float(low.tail(5).min()), 2)
        danger_price = min(ma20_val, support_5d)
        
        price_trend_20d = float(close.iloc[-1] - close.iloc[-20])
        flow_trend_20d = float(df['cum_money_flow'].iloc[-1] - df['cum_money_flow'].iloc[-20])
        if price_trend_20d > 0 and flow_trend_20d < 0:
            divergence_tag = "⚠️ ตรวจพบ Bearish Divergence: ราคาทำจุดสูงสุดใหม่ แต่เม็ดเงินจริงแอบไหลออก (ระวังการเทขายทุบตลาด)"
        elif price_trend_20d < 0 and flow_trend_20d > 0:
            divergence_tag = "✨ ตรวจพบ Bullish Divergence: ราคาย่อตัวลง แต่มีเม็ดเงินสถาบันแอบเข้าสะสมของ (ลุ้นดีดตัวกลับรอบใหญ่)"
        else:
            divergence_tag = "〰️ สภาพคล่องและทิศทางเม็ดเงินสอดคล้องกับแนวโน้มราคาตามปกติ"
        
        if d1_buy_pct >= 58 and w1_buy_pct >= 54:
            market_state = "🚀 แรงซื้อครอบงำตลาด (Strong Bullish Accumulation)"
            state_desc = "กระแสเงินไหลเข้าสะสมต่อเนื่อง โมเมนตัมฝั่งซื้อได้เปรียบสูงในทุกกรอบเวลา"
            state_color = "#10B981"
            danger_warning = f"⚠️ จุดระวัง: ดัชนียังแข็งแกร่ง แต่หากหลุด ${danger_price} อาจเกิดแรงขายทำกำไรระยะสั้น (Short-term Pullback)"
        elif d1_sell_pct >= 58 and w1_sell_pct >= 54:
            market_state = "📉 แรงขายเทกระจายของ (Heavy Distribution / Bearish)"
            state_desc = "กระแสเงินไหลออกหนาแน่น ยอดสะสมฝั่งขายคุมตลาดชัดเจน ควรเพิ่มความระมัดระวัง"
            state_color = "#F43F5E"
            danger_warning = f"🚨 จุดเตือนภัยวิกฤต: ยอดขายสะสมหนาแน่น หากหลุดต่ำกว่า ${danger_price} จะเกิด Panic Sell ระลอกใหญ่ ห้ามรับมีดเด็ดขาด"
        elif d1_buy_pct >= 52:
            market_state = "⏳ พักฐานสะสมแรง / ลุ้นดีดตัว (Healthy Pullback Flow)"
            state_desc = "ดัชนีมีแรงซื้อหยั่งเชิงพยุงตลาด ยอดสะสมรายสัปดาห์อยู่ในกรอบสะสมพลัง"
            state_color = "#F59E0B"
            danger_warning = f"⚠️ จุดระวัง: สังเกตแนวรับ ${danger_price} หากยืนได้มีโอกาสดีดตัวกลับรอบใหม่ แต่ถ้าหลุดจะเปลี่ยนเป็นขาลง"
        else:
            market_state = "〰️ ตลาดไซด์เวย์แกว่งตัวรอทิศทาง (Neutral / Choppy)"
            state_desc = "แรงซื้อและแรงขายใกล้เคียงกัน ดัชนีแกว่งตัวในกรอบแคบเพื่อรอปัจจัยหนุนใหม่"
            state_color = "#94A3B8"
            danger_warning = f"⚠️ จุดระวัง: กรอบราคาผันผวน ระวังโดนดัก Stop Loss รอเบรกเอาท์ชัดเจนก่อนเข้าเทรด"
            
        m_data = {
            'symbol': index_symbol, 'name': index_name, 'price': latest_price, 'chg_pct': chg_pct,
            'd1_buy_pct': d1_buy_pct, 'd1_sell_pct': d1_sell_pct, 'd1_winner': d1_winner,
            'w1_buy_pct': w1_buy_pct, 'w1_sell_pct': w1_sell_pct, 'w1_winner': w1_winner,
            'm1_buy_pct': m1_buy_pct, 'm1_sell_pct': m1_sell_pct, 'm1_winner': m1_winner,
            'latest_cmf': latest_cmf, 'divergence_tag': divergence_tag,
            'market_state': market_state, 'state_desc': state_desc, 'state_color': state_color,
            'danger_price': danger_price, 'danger_warning': danger_warning, 'date': df.index[-1].strftime('%d/%m/%Y')
        }
        return m_data, df
    except Exception:
        return None, None

def create_market_flow_dual_chart(df, index_name):
    if df is None or df.empty:
        return None
    try:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=[f"📈 กราฟราคา ETF {index_name} (หน่วย: USD $)", "🌊 กราฟเส้นเม็ดเงินสะสมสถาบัน (หน่วย: ล้านดอลลาร์ $M)"]
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='ราคา'
        ), row=1, col=1)

        ma20 = df['Close'].rolling(20).mean()
        ma50 = df['Close'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(color='#38BDF8', width=1.3), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ma50, line=dict(color='#FB923C', width=1.3), name='MA50'), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['cum_money_flow'],
            mode='lines',
            line=dict(color='#38BDF8', width=2),
            fill='tozeroy',
            fillcolor='rgba(56, 189, 248, 0.15)',
            name='เม็ดเงินสะสม (M$)'
        ), row=2, col=1)

        fig.update_yaxes(title_text="ราคา ETF ($)", row=1, col=1)
        fig.update_yaxes(title_text="เม็ดเงิน ($M)", row=2, col=1)

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            margin=dict(l=6, r=10, t=28, b=6),
            height=440,
            showlegend=False
        )
        return fig
    except Exception:
        return None

# ================= 6. ฟังก์ชันดึงข่าวสารเศรษฐกิจมหภาค (Macro News) =================
@st.cache_data(ttl=900, show_spinner=False)
def get_macro_market_news():
    results = []
    feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,SPY,QQQ&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F,CL=F,DX-Y.NYB&region=US&lang=en-US"
    ]
    for url in feeds:
        try:
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3.5)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall('./channel/item')[:4]:
                    t_node, l_node, p_node = item.find('title'), item.find('link'), item.find('pubDate')
                    raw_title = t_node.text if t_node is not None else ""
                    raw_link = l_node.text if l_node is not None else "#"
                    raw_pub = p_node.text[:16] if p_node is not None else ""
                    if raw_title and not any(r['link'] == raw_link for r in results):
                        title_th = translate_text_to_thai(raw_title)
                        results.append({
                            'title': title_th if title_th else raw_title,
                            'title_en': raw_title,
                            'link': raw_link,
                            'time': raw_pub
                        })
        except Exception:
            pass
        if len(results) >= 8: break
    return results

# ================= 7. ฐานข้อมูลหุ้นหลัก =================
@st.cache_data(ttl=86400)
def get_us_stock_directory(scope="TOP500"):
    master_directory = [
        {'ticker': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'เซมิคอนดักเตอร์ AI', 'exchange': 'NASDAQ'},
        {'ticker': 'TSM', 'name': 'Taiwan Semiconductor Manufacturing Co.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'ผลิตชิปเซมิคอนดักเตอร์', 'exchange': 'NYSE'},
        {'ticker': 'AAPL', 'name': 'Apple Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'อุปกรณ์สื่อสารและคอมพิวเตอร์', 'exchange': 'NASDAQ'},
        {'ticker': 'MSFT', 'name': 'Microsoft Corporation', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'ซอฟต์แวร์และคลาวด์', 'exchange': 'NASDAQ'},
        {'ticker': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': '📡 สื่อสาร / โทรคมนาคม & บันเทิง', 'industry': 'เสิร์ชเอนจิ้นและสื่อดิจิทัล', 'exchange': 'NASDAQ'},
        {'ticker': 'META', 'name': 'Meta Platforms, Inc.', 'sector': '📡 สื่อสาร / โทรคมนาคม & บันเทิง', 'industry': 'โซเชียลมีเดียและเมตาเวิร์ส', 'exchange': 'NASDAQ'},
        {'ticker': 'AMZN', 'name': 'Amazon.com, Inc.', 'sector': '🛍️ สินค้าฟุ่มเฟือย / ค้าปลีก & ยานยนต์', 'industry': 'อีคอมเมิร์ซและคลาวด์', 'exchange': 'NASDAQ'},
        {'ticker': 'AMD', 'name': 'Advanced Micro Devices, Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'โปรเซสเซอร์และกราฟิกการ์ด', 'exchange': 'NASDAQ'},
        {'ticker': 'PLTR', 'name': 'Palantir Technologies Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'แพลตฟอร์มวิเคราะห์ข้อมูล AI', 'exchange': 'NYSE'},
        {'ticker': 'MRVL', 'name': 'Marvell Technology, Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'โครงสร้างพื้นฐานเซมิคอนดักเตอร์', 'exchange': 'NASDAQ'},
        {'ticker': 'ARM', 'name': 'Arm Holdings plc', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'สถาปัตยกรรมชิปและซีพียู', 'exchange': 'NASDAQ'},
        {'ticker': 'SMCI', 'name': 'Super Micro Computer, Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'เซิร์ฟเวอร์และระบบคลาวด์ AI', 'exchange': 'NASDAQ'},
        {'ticker': 'AVGO', 'name': 'Broadcom Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'เซมิคอนดักเตอร์และซอฟต์แวร์โครงสร้าง', 'exchange': 'NASDAQ'},
        {'ticker': 'ORCL', 'name': 'Oracle Corporation', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'ฐานข้อมูลและคลาวด์', 'exchange': 'NYSE'},
        {'ticker': 'CRM', 'name': 'Salesforce, Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'ซอฟต์แวร์บริหารลูกค้าสัมพันธ์', 'exchange': 'NYSE'},
        {'ticker': 'ADBE', 'name': 'Adobe Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'ซอฟต์แวร์สร้างสรรค์ดิจิทัล', 'exchange': 'NASDAQ'},
        {'ticker': 'QCOM', 'name': 'QUALCOMM Incorporated', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'ชิปสื่อสารไร้สาย 5G', 'exchange': 'NASDAQ'},
        {'ticker': 'INTC', 'name': 'Intel Corporation', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'ผลิตชิปประมวลผล', 'exchange': 'NASDAQ'},
        {'ticker': 'TSLA', 'name': 'Tesla, Inc.', 'sector': '🛍️ สินค้าฟุ่มเฟือย / ค้าปลีก & ยานยนต์', 'industry': 'ยานยนต์ไฟฟ้าและพลังงาน', 'exchange': 'NASDAQ'},
        {'ticker': 'RKLB', 'name': 'Rocket Lab USA, Inc.', 'sector': '🏭 อุตสาหกรรม / อวกาศ & ขนส่ง', 'industry': 'เทคโนโลยีปล่อยจรวดและอวกาศ', 'exchange': 'NASDAQ'},
        {'ticker': 'JPM', 'name': 'JPMorgan Chase & Co.', 'sector': '🏦 การเงิน / ธนาคาร & ประกันภัย', 'industry': 'ธนาคารพาณิชย์ระดับโลก', 'exchange': 'NYSE'},
        {'ticker': 'V', 'name': 'Visa Inc.', 'sector': '🏦 การเงิน / ธนาคาร & ประกันภัย', 'industry': 'เครือข่ายการชำระเงินดิจิทัล', 'exchange': 'NYSE'},
        {'ticker': 'MA', 'name': 'Mastercard Incorporated', 'sector': '🏦 การเงิน / ธนาคาร & ประกันภัย', 'industry': 'บริการชำระเงินระดับโลก', 'exchange': 'NYSE'},
        {'ticker': 'LLY', 'name': 'Eli Lilly and Company', 'sector': '🏥 สุขภาพ / การแพทย์ & ยา', 'industry': 'เวชภัณฑ์และยารักษาโรค', 'exchange': 'NYSE'},
        {'ticker': 'UNH', 'name': 'UnitedHealth Group Incorporated', 'sector': '🏥 สุขภาพ / การแพทย์ & ยา', 'industry': 'ประกันสุขภาพและบริการทางการแพทย์', 'exchange': 'NYSE'},
        {'ticker': 'XOM', 'name': 'Exxon Mobil Corporation', 'sector': '⚡ พลังงาน / น้ำมัน & ก๊าซ', 'industry': 'สำรวจและผลิตน้ำมัน & ก๊าซธรรมชาติ', 'exchange': 'NYSE'},
        {'ticker': 'WMT', 'name': 'Walmart Inc.', 'sector': '🛒 สินค้าอุปโภคบริโภคจำเป็น', 'industry': 'ค้าปลีกและซูเปอร์เซ็นเตอร์', 'exchange': 'NYSE'},
        {'ticker': 'COST', 'name': 'Costco Wholesale Corporation', 'sector': '🛒 สินค้าอุปโภคบริโภคจำเป็น', 'industry': 'คลังสินค้าสมาชิกค้าปลีก', 'exchange': 'NASDAQ'},
        {'ticker': 'MSTR', 'name': 'MicroStrategy Incorporated', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'ซอฟต์แวร์องค์กรและสินทรัพย์บิตคอยน์', 'exchange': 'NASDAQ'},
        {'ticker': 'COIN', 'name': 'Coinbase Global, Inc.', 'sector': '🏦 การเงิน / ธนาคาร & ประกันภัย', 'industry': 'แพลตฟอร์มซื้อขายคริปโทเคอร์เรนซี', 'exchange': 'NASDAQ'},
        {'ticker': 'HOOD', 'name': 'Robinhood Markets, Inc.', 'sector': '🏦 การเงิน / ธนาคาร & ประกันภัย', 'industry': 'แอปพลิเคชันการลงทุนและเทรด', 'exchange': 'NASDAQ'},
        {'ticker': 'AAOI', 'name': 'Applied Optoelectronics, Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'อุปกรณ์ไฟเบอร์ออปติกและเลเซอร์', 'exchange': 'NASDAQ'},
        {'ticker': 'CRWV', 'name': 'CoreWeave, Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'คลาวด์คอมพิวติ้งสำหรับ AI', 'exchange': 'NASDAQ'},
        {'ticker': 'RXT', 'name': 'Rackspace Technology, Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'บริการมัลติคลาวด์และโฮสติ้ง', 'exchange': 'NASDAQ'},
        {'ticker': 'BZAI', 'name': 'Blaize Holdings, Inc.', 'sector': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์', 'industry': 'โปรเซสเซอร์ Edge AI', 'exchange': 'NASDAQ'}
    ]
    return master_directory

# ================= 8. ฟังก์ชันวิเคราะห์เทคนิคอลรายตัว =================
def calculate_single_swing_snr(df, latest_close):
    n = len(df)
    window_n = min(n, 120)
    df_wave = df.iloc[-window_n:]
    highs, lows = df_wave['high'].values, df_wave['low'].values
    wave_high, wave_low = float(np.max(highs)), float(np.min(lows))
    wave_range = max(1e-4, wave_high - wave_low)

    fib_236 = wave_high - 0.236 * wave_range
    fib_382 = wave_high - 0.382 * wave_range
    fib_500 = wave_high - 0.500 * wave_range
    fib_618 = wave_high - 0.618 * wave_range

    recent_15d_low = float(np.min(lows[-15:]))
    recent_45d_low = float(np.min(lows[-45:]))

    if recent_15d_low < latest_close * 0.995 and recent_15d_low > latest_close * 0.85: s1 = recent_15d_low
    elif fib_500 < latest_close * 0.995 and fib_500 > latest_close * 0.88: s1 = fib_500
    elif fib_618 < latest_close * 0.995: s1 = fib_618
    else: s1 = latest_close * 0.95

    if recent_45d_low < s1 * 0.96 and recent_45d_low > wave_low * 1.15: s2 = recent_45d_low
    elif fib_618 < s1 * 0.96: s2 = fib_618
    else: s2 = s1 * 0.89

    s3 = wave_low if wave_low < s2 * 0.90 else s2 * 0.75
    if s2 >= s1: s2 = s1 * 0.90
    if s3 >= s2: s3 = s2 * 0.75

    r4 = wave_high
    cand_resists = [fib_500, fib_382, fib_236]
    valid_resists = sorted([r for r in cand_resists if r > latest_close * 1.015 and r < r4 * 0.985])

    if len(valid_resists) >= 3: r1, r2, r3 = valid_resists[0], valid_resists[1], valid_resists[2]
    elif len(valid_resists) == 2: r1, r2 = valid_resists[0], valid_resists[1]; r3 = r2 + (r4 - r2) * 0.50
    elif len(valid_resists) == 1: r1 = valid_resists[0]; r2 = r1 + (r4 - r1) * 0.35; r3 = r1 + (r4 - r1) * 0.70
    else: r1 = latest_close * 1.06; r2 = latest_close * 1.15; r3 = latest_close * 1.25

    decimals = 4 if latest_close < 2.0 else 2
    return round(s1, decimals), round(s2, decimals), round(s3, decimals), round(r1, decimals), round(r2, decimals), round(r3, decimals), round(r4, decimals)

@st.cache_data(ttl=3600, show_spinner=False)
def check_ma_snr_combo(item_input, info_mode=False):
    try:
        if isinstance(item_input, dict):
            ticker = item_input.get('ticker', '')
            hint_name = item_input.get('name', ticker)
            hint_sector = item_input.get('sector', item_input.get('type', 'Global Market'))
            hint_industry = item_input.get('industry', 'สินทรัพย์การเงิน')
            hint_exchange = item_input.get('exchange', 'Global Market')
        else:
            ticker = str(item_input).strip().upper()
            hint_name = ticker
            hint_sector = 'Global Market'
            hint_industry = 'สินทรัพย์การเงิน'
            hint_exchange = 'Global Market'

        df, detected_exchange, meta_name = fetch_stock_history_dual(ticker)
        if df is None or df.empty or len(df) < 15:
            return None, None

        final_exchange = detected_exchange if detected_exchange != "Global Market" else hint_exchange
        final_company_name = hint_name if hint_name != ticker else (meta_name if meta_name else ticker)

        latest_close = float(df['close'].iloc[-1])
        latest_open = float(df['open'].iloc[-1])
        is_today_green = latest_close >= latest_open

        fast_ma = df['close'].rolling(20).mean()
        slow_ma = df['close'].rolling(50).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        loss_safe = loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + (gain / loss_safe)))
        latest_rsi = round(float(rsi_series.fillna(100.0).iloc[-1]), 2)

        s1, s2, s3, r1, r2, r3, r4 = calculate_single_swing_snr(df, latest_close)

        recent_8d_high = float(df['high'].tail(8).max())
        drop_8d_pct = ((latest_close - recent_8d_high) / recent_8d_high) * 100
        recent_8d_low = float(df['low'].tail(8).min())
        bounce_8d_pct = ((latest_close - recent_8d_low) / recent_8d_low) * 100 if recent_8d_low > 0 else 0.0

        is_above_ma20 = latest_close >= (fast_ma.iloc[-1] if pd.notna(fast_ma.iloc[-1]) else latest_close)
        is_above_ma50 = latest_close >= (slow_ma.iloc[-1] if pd.notna(slow_ma.iloc[-1]) else latest_close)
        is_ma_bull = (fast_ma.iloc[-1] >= slow_ma.iloc[-1]) if (pd.notna(fast_ma.iloc[-1]) and pd.notna(slow_ma.iloc[-1])) else False

        bull_score = 0
        if is_above_ma20: bull_score += 30
        if is_above_ma50: bull_score += 25
        if is_ma_bull: bull_score += 20
        if 50 <= latest_rsi <= 72: bull_score += 15
        elif 40 <= latest_rsi < 50: bull_score += 5
        if drop_8d_pct > -8: bull_score += 10
        bullish_pct = min(96.0, max(5.0, round(bull_score * 0.95 + 4.0, 1)))

        dist_s1_pct = ((latest_close - s1) / s1) * 100

        if (not is_above_ma20 and not is_above_ma50 and latest_rsi < 45) or drop_8d_pct <= -15.0:
            trend_status = "DOWNTREND"
            status_text = "📉 ลงแรง / ขาลงชัดเจน (ห้ามรับมีด)"
            status_box_class = "status-banner-downtrend"
            badge_class = "badge-board-downtrend"
            badge_label = f"📉 ขาลงชัดเจน: {100.0 - bullish_pct:.1f}%"
            status_desc = f"⚠️ ราคาหลุดเส้นค่าเฉลี่ยหลัก (ย่อตัวจากยอด 8 วัน {drop_8d_pct:.2f}%) โครงสร้างเสียเปรียบ ยังไม่ควรเข้าซื้อ"

        elif (is_above_ma50 or is_ma_bull or bounce_8d_pct >= 8.0) and (drop_8d_pct <= -3.0 or not is_today_green) and latest_rsi >= 40:
            trend_status = "PULLBACK"
            status_text = "⏳ ย่อพักฐาน (Healthy Pullback)"
            status_box_class = "status-banner-pullback"
            badge_class = "badge-board-pullback"
            badge_label = f"⏳ ย่อพักฐาน: {bullish_pct}%"
            status_desc = f"🔄 อยู่ในแนวโน้มใหญ่ขาขึ้น แต่กำลังย่อตัวพักฐานตามรอบ (ย่อจากยอด 8 วัน {drop_8d_pct:.2f}%) เพื่อสะสมแรง"

        elif dist_s1_pct <= 4.5 and bounce_8d_pct <= 5.5 and latest_close >= s1 * 0.98:
            trend_status = "BUY_SUPPORT"
            status_text = f"🎯 ช้อนแนวรับ (เด้งจากฐาน +{bounce_8d_pct:.2f}%)"
            status_box_class = "status-banner-support"
            badge_class = "badge-board-support"
            badge_label = f"🎯 ช้อนแนวรับ: {bullish_pct}%"
            status_desc = f"🛡️ ราคาอยู่ในโซนแนวรับสำคัญและเริ่มมีแรงดีดกลับตัว (+{bounce_8d_pct:.2f}%) เหมาะสะสมไม้ 1"

        elif is_above_ma20 and is_above_ma50 and latest_rsi >= 50 and drop_8d_pct > -3.0:
            trend_status = "UPTREND"
            status_text = "🚀 ขาขึ้นแข็งแกร่ง (Strong Uptrend)"
            status_box_class = "status-banner-uptrend"
            badge_class = "badge-board-uptrend"
            badge_label = f"🚀 ขาขึ้นแข็งแกร่ง: {bullish_pct}%"
            status_desc = f"✨ ราคายืนเหนือเส้นแนวโน้มหลักทุกเส้น โมเมนตัมขาขึ้นสมบูรณ์ (ดีดตัวจากฐานล่าสุด +{bounce_8d_pct:.2f}%)"

        else:
            trend_status = "SIDEWAYS"
            status_text = "〰️ สะสมแรง / ไซด์เวย์"
            status_box_class = "status-banner-sideways"
            badge_class = "badge-board-sideways"
            badge_label = f"〰️ สะสมแรง/ไซด์เวย์: {bullish_pct}%"
            status_desc = f"📦 ราคาแกว่งตัวสร้างฐานในกรอบแคบ ยังไม่มีทิศทางชัดเจน รอการเบรกเอาท์"

        dist_from_sup = ((latest_close - s1) / s1) * 100
        pat_name, pat_score = calculate_ai_pattern_match(df.tail(60))
        vol_val = df['volume'].iloc[-1] if 'volume' in df.columns else 0

        res_data = {
            'Ticker': ticker,
            'longNameEn': final_company_name,
            'sectorTh': hint_sector,
            'industryTh': hint_industry,
            'Exchange': final_exchange,
            'Price ($)': latest_close,
            'Support 1 ($)': s1,
            'Support 2 ($)': s2,
            'Support 3 ($)': s3,
            'Resist 1 ($)': r1,
            'Resist 2 ($)': r2,
            'Resist 3 ($)': r3,
            'Resist 4 ($)': r4,
            'Dist_Sup (%)': f'{dist_from_sup:+.2f}%',
            'RSI': latest_rsi,
            'Volume': f"{vol_val:,.0f}",
            'Date': df.index[-1].strftime('%Y-%m-%d'),
            'pattern_name': pat_name,
            'pattern_score': pat_score,
            'bullish_pct': bullish_pct,
            'trend_status': trend_status,
            'status_text': status_text,
            'status_box_class': status_box_class,
            'badge_label': badge_label,
            'badge_class': badge_class,
            'status_desc': status_desc,
            'drop_8d_pct': f'{drop_8d_pct:.2f}%',
            'bounce_8d_pct': f'+{bounce_8d_pct:.2f}%'
        }

        co_info = get_company_info_and_holders(ticker)
        if co_info:
            if co_info.get('longNameEn') and co_info['longNameEn'] != ticker:
                res_data['longNameEn'] = co_info['longNameEn']
            if co_info.get('sectorTh') and co_info['sectorTh'] != 'N/A':
                res_data['sectorTh'] = co_info['sectorTh']
            if co_info.get('industryTh') and co_info['industryTh'] != 'N/A':
                res_data['industryTh'] = co_info['industryTh']
            res_data['summaryTh'] = co_info.get('summaryTh', 'N/A')
            res_data.update(co_info)

        if not info_mode:
            if not (trend_status in ["UPTREND", "PULLBACK", "BUY_SUPPORT"]):
                return None, df

        return res_data, df
    except Exception:
        pass
    return None, None

def create_ta_chart(df, ticker, res_data):
    if df is None or df.empty or res_data is None:
        return None
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='ราคา'
        )])
        fast_ma = df['close'].rolling(20).mean()
        slow_ma = df['close'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=fast_ma, line=dict(color='#38BDF8', width=1.2), name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=slow_ma, line=dict(color='#FB923C', width=1.2), name='MA50'))

        for key, color, ay_pos in [('Support 1 ($)', '#22C55E', -12), ('Support 2 ($)', '#16A34A', 12), ('Support 3 ($)', '#15803D', -12)]:
            if key in res_data and res_data[key] is not None:
                val = res_data[key]
                fig.add_shape(type="line", x0=df.index[0], y0=val, x1=df.index[-1], y1=val, line=dict(color=color, width=1.6, dash='dash'))
                fig.add_annotation(x=df.index[-1], y=val, text=f"{key.replace(' ($)', '')}: ${val}", bgcolor=color, font=dict(color="white", size=9), xanchor="left", ax=8, ay=ay_pos)

        for key, color, ay_pos in [('Resist 1 ($)', '#EF4444', -12), ('Resist 2 ($)', '#F97316', 12), ('Resist 3 ($)', '#EAB308', -12), ('Resist 4 ($)', '#991B1B', 12)]:
            if key in res_data and res_data[key] is not None:
                val = res_data[key]
                fig.add_shape(type="line", x0=df.index[0], y0=val, x1=df.index[-1], y1=val, line=dict(color=color, width=1.6, dash='dash'))
                fig.add_annotation(x=df.index[-1], y=val, text=f"{key.replace(' ($)', '')}: ${val}", bgcolor=color, font=dict(color="white", size=9), xanchor="left", ax=8, ay=ay_pos)

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', margin=dict(l=6, r=65, t=10, b=6), height=340, dragmode='pan', yaxis_title="ราคา", showlegend=False)
        return fig
    except Exception:
        return None

# ================= 9. ฟังก์ชันช่วยเรนเดอร์ UI รายละเอียด =================
def render_analysis_view(res, raw_df, df_profit, news_items, single_ticker, is_forex=False):
    company_full_name = res.get("longNameEn", single_ticker)
    sector_desc = res.get("sectorTh", "N/A")
    industry_desc = res.get("industryTh", "N/A")
    exchange_desc = res.get("Exchange", "Global Market")

    st.markdown(f'<p class="company-header">{single_ticker} : {company_full_name} <span class="price-badge badge-market">🏛️ {exchange_desc}</span></p>', unsafe_allow_html=True)
    st.markdown(f'<div class="sector-badge">🏷️ ประเภท: {sector_desc} | กลุ่ม: {industry_desc}</div>', unsafe_allow_html=True)
    
    box_css = res.get('status_box_class', 'status-banner-sideways')
    st.markdown(f"""
    <div class="status-banner {box_css}">
        <div class="status-title-text">{res.get('status_text', '')}</div>
        <div class="status-desc-text">{res.get('status_desc', '')} | ข้อมูล ณ วันที่: {res.get('Date', '')} (⚡ โหลดความเร็วสูง)</div>
    </div>
    """, unsafe_allow_html=True)

    if raw_df is not None:
        st.markdown(f"#### {UI_LANG_MAP['chart_title_single']}")
        st.markdown(f'<div class="chart-header-badge">{single_ticker} ({exchange_desc}) | ล่าสุด: {res.get("Price ($)", 0)} (RSI: {res.get("RSI", 0)})</div>', unsafe_allow_html=True)
        fig = create_ta_chart(raw_df, single_ticker, res)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"chart_{single_ticker}_{is_forex}")

    st.markdown("---")
    st.markdown(f"#### {UI_LANG_MAP['analysis_title']}")
    
    badge_class = res.get('badge_class', 'badge-board-sideways')
    badge_label = res.get('badge_label', '〰️ สะสมแรง')
    pat_name = res.get('pattern_name', 'สร้างฐานสะสมกำลัง.png')
    pat_score = res.get('pattern_score', 75.0)

    st.markdown(f"""
    <div class="compact-board">
        <div class="price-banner">
            <div class="price-val-box">
                <span style="font-size:0.8rem; color:#94A3B8; font-weight:600;">💰 ราคา:</span>
                <span class="price-main">{res.get('Price ($)', 0)}</span>
            </div>
            <div class="price-badge-group">
                <span class="price-badge badge-market">🏛️ {exchange_desc}</span>
                <span class="price-badge {badge_class}">{badge_label}</span>
                <span class="price-badge badge-ai-box">🤖 AI Pattern: {pat_name} ({pat_score}%)</span>
                <span class="price-badge badge-rsi">RSI: {res.get('RSI', 0)}</span>
                <span class="price-badge badge-dist">ห่างรับ 1: {res.get('Dist_Sup (%)', '0%')}</span>
            </div>
        </div>
        <div class="snr-grid">
            <div class="snr-card" style="border-left: 3px solid #22C55E;">
                <div class="snr-card-title c-green">🛡️ แนวรับ (Support)</div>
                <div class="snr-row"><span class="snr-lbl">รับ 1 (สวิงใกล้สุด)</span><span class="snr-num c-green">{res.get('Support 1 ($)', 0)}</span></div>
                <div class="snr-row"><span class="snr-lbl">รับ 2 (ฐานหลัก)</span><span class="snr-num c-lightgreen">{res.get('Support 2 ($)', 0)}</span></div>
                <div class="snr-row"><span class="snr-lbl">รับ 3 (โครงสร้างใหญ่)</span><span class="snr-num c-lightgreen">{res.get('Support 3 ($)', 0)}</span></div>
            </div>
            <div class="snr-card" style="border-left: 3px solid #EF4444;">
                <div class="snr-card-title c-red">⚡ แนวต้าน (Resistance)</div>
                <div class="snr-row"><span class="snr-lbl">ต้าน 1</span><span class="snr-num c-red">{res.get('Resist 1 ($)', 0)}</span></div>
                <div class="snr-row"><span class="snr-lbl">ต้าน 2</span><span class="snr-num c-orange">{res.get('Resist 2 ($)', 0)}</span></div>
                <div class="snr-row"><span class="snr-lbl">ต้าน 3</span><span class="snr-num c-yellow">{res.get('Resist 3 ($)', 0)}</span></div>
                <div class="snr-row"><span class="snr-lbl">ต้าน 4 (สูงสุด)</span><span class="snr-num c-darkred">{res.get('Resist 4 ($)', 0)}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎯 กลยุทธ์แบ่งไม้เข้าซื้อ & ประเมินความแข็งแรงของแนวรับ")
    curr_p = res.get('Price ($)', 1.0)
    dist_s1 = ((res.get('Support 1 ($)', 0) - curr_p) / curr_p) * 100
    dist_s2 = ((res.get('Support 2 ($)', 0) - curr_p) / curr_p) * 100
    dist_s3 = ((res.get('Support 3 ($)', 0) - curr_p) / curr_p) * 100

    st.markdown(f"""
    <div class="strategy-card" style="border-left: 4px solid #22C55E;">
        <div class="strat-header">
            <div><span class="strat-title">🛡️ แนวรับ 1 (สวิงโลว์ใกล้สุด)</span><span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">({dist_s1:+.2f}%)</span></div>
            <span class="strat-price" style="color:#22C55E;">{res.get('Support 1 ($)', 0)}</span>
        </div>
        <div class="strat-body">
            <div><span class="strat-sub">ความแข็งแรง:</span> <span class="strat-val">⭐⭐ ปานกลาง</span></div>
            <div><span class="strat-sub">กลยุทธ์:</span> <span class="strat-val c-green">25% (ไม้หยั่งเชิง / ดูแรงเด้ง)</span></div>
        </div>
    </div>

    <div class="strategy-card" style="border-left: 4px solid #16A34A;">
        <div class="strat-header">
            <div><span class="strat-title">🛡️ แนวรับ 2 (ฐานสะสมหลัก)</span><span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">({dist_s2:+.2f}%)</span></div>
            <span class="strat-price" style="color:#4ADE80;">{res.get('Support 2 ($)', 0)}</span>
        </div>
        <div class="strat-body">
            <div><span class="strat-sub">ความแข็งแรง:</span> <span class="strat-val">⭐⭐⭐⭐ แข็งแกร่ง</span></div>
            <div><span class="strat-sub">กลยุทธ์:</span> <span class="strat-val c-lightgreen">35% (ไม้หลักสะสมของ)</span></div>
        </div>
    </div>

    <div class="strategy-card" style="border-left: 4px solid #15803D;">
        <div class="strat-header">
            <div><span class="strat-title">🛡️ แนวรับ 3 (ฐานโครงสร้างใหญ่)</span><span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">({dist_s3:+.2f}%)</span></div>
            <span class="strat-price" style="color:#86EFAC;">{res.get('Support 3 ($)', 0)}</span>
        </div>
        <div class="strat-body">
            <div><span class="strat-sub">ความแข็งแรง:</span> <span class="strat-val">⭐⭐⭐⭐⭐ แข็งแกร่งมาก</span></div>
            <div><span class="strat-sub">กลยุทธ์:</span> <span class="strat-val" style="color:#86EFAC;">40% (ไม้สะสมลึก / กลับตัวใหญ่)</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not is_forex:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📰 ข่าวสารล่าสุด & ปัจจัยกระทบ (แปลไทยอัตโนมัติ)")
        if news_items:
            for news in news_items:
                pub_time = f" | เผยแพร่เมื่อ: {news['time']}" if news['time'] else ""
                st.markdown(f"""
                <div class="news-card">
                    <a class="news-title" href="{news['link']}" target="_blank">📌 {news['title']}</a>
                    <div class="news-meta">แหล่งข่าว: {news['publisher']}{pub_time}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"ℹ️ ไม่พบหัวข้อข่าวสำคัญล่าสุดสำหรับหุ้น {single_ticker}")

        st.markdown("#### 💰 กำไรสุทธิ 3 ไตรมาสล่าสุด")
        if df_profit is not None:
            c_table, c_chart = st.columns(2)
            with c_table:
                st.dataframe(df_profit, use_container_width=True, hide_index=True, height=125)
            with c_chart:
                bar_colors = ['#22C55E' if v >= 0 else '#EF4444' for v in df_profit['Net Income (M$)']]
                fig_profit = go.Figure(data=[go.Bar(
                    x=df_profit['Quarter End'], y=df_profit['Net Income (M$)'], marker_color=bar_colors
                )])
                fig_profit.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=125, template='plotly_dark', xaxis_title="", yaxis_title="M$")
                st.plotly_chart(fig_profit, use_container_width=True, config={'displayModeBar': False}, key=f"chart_profit_{single_ticker}")
        else:
            st.info("ℹ️ ไม่พบข้อมูลงบกำไรสุทธิย้อนหลังสำหรับสัญลักษณ์นี้")

        st.markdown("---")
        summary_text = res.get('summaryTh', 'N/A')
        shares_tot = res.get('sharesOutstanding', 'N/A')
        inst_pct = res.get('institutionalHeld', 'N/A')
        insider_pct = res.get('insiderHeld', 'N/A')
        retail_pct = res.get('retailHeld', 'N/A')

        with st.expander(UI_LANG_MAP.get('expander_business_summary', "📖 สรุปธุรกิจ & โครงสร้างผู้ถือหุ้น (แปลไทยอัตโนมัติ)"), expanded=True):
            st.markdown(f"""
            <div class="fin-card">
                <b style="color: #60A5FA; font-size: 0.88rem;">📊 โครงสร้างผู้ถือหุ้น & ข้อมูลบริษัท:</b>
                <div style="color: #F8FAFC; line-height: 1.7; margin-top: 4px; font-size: 0.82rem;">
                • ตลาดซื้อขาย: <b style="color: #38BDF8;">{exchange_desc}</b><br>
                • กลุ่มธุรกิจ: <b style="color: #FCD34D;">{sector_desc}</b><br>
                • อุตสาหกรรมย่อย: <b style="color: #E2E8F0;">{industry_desc}</b><br>
                • จำนวนหุ้นทั้งหมด: <b style="color: #FFFFFF;">{shares_tot} หุ้น</b><br>
                • สถาบันถือครอง: <b style="color: #38BDF8;">{inst_pct}</b><br>
                • ผู้บริหาร/Insider ถือครอง: <b style="color: #FBBF24;">{insider_pct}</b><br>
                • รายย่อยและอื่นๆ ถือครอง: <b style="color: #34D399;">{retail_pct}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if summary_text != 'N/A':
                 st.markdown(f'<div class="biz-summary">{summary_text}</div>', unsafe_allow_html=True)

# ================= 10. ส่วนแสดงผล UI หน้าจอ (5 แท็บสมบูรณ์) =================
tab_market, tab1, tab2, tab3, tab_news = st.tabs([
    UI_LANG_MAP['tab_market_flow'],
    UI_LANG_MAP['tab_search_ticker'],
    UI_LANG_MAP['tab_scan_market'],
    UI_LANG_MAP['tab_forex'],
    UI_LANG_MAP['tab_macro_news']
])

# --- TAB 1: ทิศทางตลาด Nasdaq & S&P 500 (Buy-Sell Flow + Live Dual Charts) ---
with tab_market:
    st.markdown("### 🏛️ วิเคราะห์กระแสเงินและทิศทางตลาด (Nasdaq & S&P 500 Flow)")
    st.caption("คำนวณสัดส่วนแรงซื้อ vs แรงขายสะสม, กราฟเส้นเม็ดเงินสถาบัน (Cumulative Flow) และ AI Divergence")
    
    if st.button("🔄 อัปเดตข้อมูลกระแสเงินสดตลาด", key="btn_refresh_market_flow"):
        calculate_market_flow_advanced.clear()
        st.rerun()

    with st.spinner("⏳ กำลังคำนวณกระแสเงินสดและกราฟราคาดัชนี Nasdaq (QQQ) และ S&P 500 (SPY)..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_qqq = executor.submit(calculate_market_flow_advanced, "QQQ", "Nasdaq 100 Index (QQQ)")
            f_spy = executor.submit(calculate_market_flow_advanced, "SPY", "S&P 500 Index (SPY)")
            data_qqq, df_qqq = f_qqq.result()
            data_spy, df_spy = f_spy.result()

    col_m1, col_m2 = st.columns(2)
    for col, m_data, m_df in zip([col_m1, col_m2], [data_qqq, data_spy], [df_qqq, df_spy]):
        if m_data:
            with col:
                html_flow_card = f"""<div class="market-flow-card">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
<span style="font-size:1.15rem; font-weight:800; color:#60A5FA;">📈 {m_data['name']}</span>
<span style="font-size:1.1rem; font-weight:800; color:{'#10B981' if m_data['chg_pct'] >= 0 else '#F43F5E'};">${m_data['price']} ({m_data['chg_pct']:+.2f}%)</span>
</div>
<div style="font-size:0.9rem; font-weight:bold; color:{m_data['state_color']}; margin-bottom:4px;">{m_data['market_state']}</div>
<div style="font-size:0.78rem; color:#94A3B8; margin-bottom:12px;">{m_data['state_desc']}</div>
<div style="display:flex; justify-content:space-between; font-size:0.75rem; font-weight:600; margin-bottom:4px;">
<span class="c-green">🟢 แรงซื้อวันนี้: {m_data['d1_buy_pct']}%</span>
<span class="c-red">🔴 แรงขายวันนี้: {m_data['d1_sell_pct']}%</span>
</div>
<div class="flow-meter-container">
<div class="flow-buy-bar" style="width:{m_data['d1_buy_pct']}%;">{m_data['d1_buy_pct']}%</div>
<div class="flow-sell-bar" style="width:{m_data['d1_sell_pct']}%;">{m_data['d1_sell_pct']}%</div>
</div>
<div class="flow-grid-3">
<div class="flow-sub-card">
<div style="font-size:0.7rem; color:#94A3B8;">รายวัน (1D)</div>
<div style="font-size:0.82rem; font-weight:800; color:{'#10B981' if m_data['d1_buy_pct']>=50 else '#F43F5E'};">{m_data['d1_winner']}</div>
<div style="font-size:0.72rem; color:#cbd5e1;">ซื้อ {m_data['d1_buy_pct']}% | ขาย {m_data['d1_sell_pct']}%</div>
</div>
<div class="flow-sub-card">
<div style="font-size:0.7rem; color:#94A3B8;">รายสัปดาห์ (1W / 5 วัน)</div>
<div style="font-size:0.82rem; font-weight:800; color:{'#10B981' if m_data['w1_buy_pct']>=50 else '#F43F5E'};">{m_data['w1_winner']}</div>
<div style="font-size:0.72rem; color:#cbd5e1;">ซื้อ {m_data['w1_buy_pct']}% | ขาย {m_data['w1_sell_pct']}%</div>
</div>
<div class="flow-sub-card">
<div style="font-size:0.7rem; color:#94A3B8;">รายเดือน (1M / 21 วัน)</div>
<div style="font-size:0.82rem; font-weight:800; color:{'#10B981' if m_data['m1_buy_pct']>=50 else '#F43F5E'};">{m_data['m1_winner']}</div>
<div style="font-size:0.72rem; color:#cbd5e1;">ซื้อ {m_data['m1_buy_pct']}% | ขาย {m_data['m1_sell_pct']}%</div>
</div>
</div>
<div class="danger-alert-box">
<b>⚠️ จุดระวัง & ระดับราคาชี้เป็นชี้ตาย:</b><br>
{m_data['danger_warning']}<br>
<b style="color:#60A5FA;">🤖 AI Flow Alert:</b> {m_data['divergence_tag']}
</div>
</div>"""
                st.markdown(html_flow_card, unsafe_allow_html=True)
                
                if m_df is not None:
                    fig_flow_chart = create_market_flow_dual_chart(m_df, m_data['name'])
                    if fig_flow_chart is not None:
                        st.plotly_chart(fig_flow_chart, use_container_width=True, config=PLOTLY_CONFIG, key=f"flow_chart_{m_data['symbol']}")


# --- TAB 2: ค้นหา & วิเคราะห์หุ้นรายตัว ---
with tab1:
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        single_ticker = st.text_input(UI_LANG_MAP['search_ticker_label'], value='NVDA').strip().upper()
    with col_in2:
        st.markdown("<div class='desktop-only-space'></div>", unsafe_allow_html=True)
        search_btn = st.button(UI_LANG_MAP['btn_analyze_single'], key="btn_search_stock")

    if search_btn and single_ticker:
        with st.spinner(UI_LANG_MAP['status_analyzing_single'].format(ticker=single_ticker)):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                f_combo = executor.submit(check_ma_snr_combo, single_ticker, True)
                f_profit = executor.submit(get_financials, single_ticker)
                f_news = executor.submit(get_stock_news, single_ticker)

                res, raw_df = f_combo.result()
                df_profit = f_profit.result()
                news_items = f_news.result()

            if res:
                render_analysis_view(res, raw_df, df_profit, news_items, single_ticker, is_forex=False)
            else:
                st.error(f"❌ ไม่พบข้อมูลสัญลักษณ์หุ้น **{single_ticker}** ในระบบ กรุณาตรวจสอบชื่อ Ticker อีกครั้ง")


# --- TAB 3: สแกนคัดหุ้นทรงสวย (ทั้งตลาด) ---
with tab2:
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน (NASDAQ, NYSE, AMEX)")
    
    if server_state["is_scanning"] and server_state.get("scan_start_time"):
        elapsed_scan = (datetime.now() - server_state["scan_start_time"]).total_seconds()
        if elapsed_scan > 120:
            server_state["is_scanning"] = False
            server_state["scan_start_time"] = None

    scan_scope = st.radio(
        "🎯 เลือกขอบเขตและจำนวนหุ้นที่จะสแกน:",
        ["⚡ หุ้นผู้นำตลาด S&P 500 & Top Tech (500 ตัวจริงครบ A-Z - สแกนเร็ว 15 วิ)",
         "🚀 หุ้น Growth & Momentum ชั้นนำ (1,000 ตัวคัดเกรด - แนะนำ 35 วิ)",
         "🌐 หุ้น Active สภาพคล่องสูงทั้งตลาด (2,500+ หุ้น - เต็มระบบ 1 นาที)"],
        index=0,
        horizontal=True
    )
    
    scope_code = "TOP500" if "500" in scan_scope else ("GROWTH1000" if "1,000" in scan_scope else "ALL")
    is_busy = server_state["is_scanning"]
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        scan_btn = st.button(UI_LANG_MAP['btn_scan_market'], disabled=is_busy, key="btn_scan_all")
    with col_btn2:
        reset_btn = st.button("🔄 ปลดล็อก & รีเซ็ตระบบ", key="btn_reset_all")

    if reset_btn:
        server_state["is_scanning"] = False
        server_state["scan_start_time"] = None
        server_state["latest_results"] = None
        server_state["latest_df"] = None
        server_state["last_scanned_at"] = None
        server_state["last_scanned_dt"] = None
        st.success("ปลดล็อกและล้างข้อมูลเรียบร้อยแล้ว")
        st.rerun()

    if scan_btn and not is_busy:
        check_ma_snr_combo.clear()
        server_state["is_scanning"] = True
        server_state["scan_start_time"] = datetime.now()
        status_text = st.empty()
        status_text.info(UI_LANG_MAP['status_preparing_tickers'])
        
        stock_directory = get_us_stock_directory(scope_code)
        total_stocks = len(stock_directory)
        progress_bar = st.progress(0)
        
        results = []
        count = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
                futures = {executor.submit(check_ma_snr_combo, item, False): item for item in stock_directory}
                for future in concurrent.futures.as_completed(futures):
                    count += 1
                    if count % 15 == 0 or count == total_stocks:
                        progress_bar.progress(count / total_stocks)
                        status_text.text(UI_LANG_MAP['status_scanning'].format(count=count, total=total_stocks, found=len(results)))
                    try:
                        res_data_found, raw_df_found = future.result()
                        if res_data_found and raw_df_found is not None:
                            results.append({'res_data': res_data_found, 'raw_df': raw_df_found})
                    except Exception:
                        pass
        finally:
            server_state["is_scanning"] = False
            server_state["scan_start_time"] = None

        status_text.empty()
        st.success(f'✅ สแกนเสร็จสิ้นจากทั้งหมด {total_stocks:,} ตัว! พบหุ้นทรงสวยเข้าเกณฑ์ {len(results)} ตัว')
        
        server_state["latest_results"] = results
        server_state["last_scanned_dt"] = datetime.now()
        server_state["last_scanned_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if results:
            all_res_data = [item['res_data'] for item in results]
            df_result_display = pd.DataFrame(all_res_data)
            cols_to_show = [
                'Ticker', 'longNameEn', 'Exchange', 'sectorTh', 'Price ($)', 'status_text', 
                'pattern_name', 'Support 1 ($)', 'Support 2 ($)', 'Support 3 ($)', 'RSI', 
                'Resist 1 ($)', 'Resist 2 ($)', 'Resist 3 ($)', 'Resist 4 ($)', 
                'Volume', 'Date'
            ]
            available_cols = [c for c in cols_to_show if c in df_result_display.columns]
            server_state["latest_df"] = df_result_display[available_cols]
        st.rerun()

    if server_state["latest_results"]:
        all_results = server_state["latest_results"]
        total_found = len(all_results)
        nasdaq_count = sum(1 for item in all_results if "NASDAQ" in item['res_data'].get('Exchange', '').upper())
        nyse_count = sum(1 for item in all_results if "NYSE" in item['res_data'].get('Exchange', '').upper())
        amex_count = sum(1 for item in all_results if "AMEX" in item['res_data'].get('Exchange', '').upper())

        if server_state.get("last_scanned_at"):
            elapsed_thai = get_time_elapsed_thai(server_state.get("last_scanned_dt"))
            st.info(f"🕒 สแกนล่าสุด: **{server_state['last_scanned_at']}**{elapsed_thai} | ทั้งหมด **{total_found}** ตัว (💻 NASDAQ: **{nasdaq_count}** | 🏛️ NYSE: **{nyse_count}** | 🏭 AMEX: **{amex_count}**)")

        st.markdown("---")
        
        col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 2])
        with col_f1:
            market_filter = st.selectbox("🏢 กรองตามตลาด:", ["ทั้งหมด (All Exchanges)", "เฉพาะ NASDAQ", "เฉพาะ NYSE", "เฉพาะ AMEX"], key="filter_market_choice")
        with col_f2:
            status_filter = st.selectbox("🎯 กรองตามสถานะ:", ["ทั้งหมด (All Statuses)", "🚀 ขาขึ้นแข็งแกร่ง", "⏳ ย่อพักฐาน", "🎯 ช้อนแนวรับ"], key="filter_status_choice")
        with col_f3:
            ticker_search_filter = st.text_input("🔍 ค้นหาชื่อหุ้นในผลลัพธ์:", value="", key="filter_search_ticker").strip().upper()

        filtered_results = []
        for item in all_results:
            r = item['res_data']
            ex_val = r.get('Exchange', '').upper()
            st_val = r.get('status_text', '')
            sym_val = r.get('Ticker', '').upper()
            name_val = r.get('longNameEn', '').upper()

            if market_filter == "เฉพาะ NASDAQ" and "NASDAQ" not in ex_val: continue
            if market_filter == "เฉพาะ NYSE" and "NYSE" not in ex_val: continue
            if market_filter == "เฉพาะ AMEX" and "AMEX" not in ex_val: continue
            if status_filter == "🚀 ขาขึ้นแข็งแกร่ง" and "ขาขึ้น" not in st_val: continue
            if status_filter == "⏳ ย่อพักฐาน" and "ย่อพักฐาน" not in st_val: continue
            if status_filter == "🎯 ช้อนแนวรับ" and "ช้อนแนวรับ" not in st_val: continue
            if ticker_search_filter and (ticker_search_filter not in sym_val and ticker_search_filter not in name_val): continue

            filtered_results.append(item)

        st.caption(f"📌 แสดงผลลัพธ์ที่ตรงตามเงื่อนไข: **{len(filtered_results)}** จากทั้งหมด **{total_found}** ตัว")
        st.subheader('📸 แกลเลอรี่กราฟหุ้นทรงสวย')
        
        if filtered_results:
            items_per_page = 6
            total_pages = max(1, (len(filtered_results) + items_per_page - 1) // items_per_page)
            page_num = st.selectbox("เลือกหน้าแสดงผลกราฟ:", range(1, int(total_pages) + 1), key="pagination_select")
            
            start_idx = (page_num - 1) * items_per_page
            end_idx = start_idx + items_per_page
            current_page_items = filtered_results[start_idx:end_idx]

            for row_idx in range(0, len(current_page_items), 2):
                cols = st.columns(2)
                for c_offset in range(2):
                    item_idx = row_idx + c_offset
                    if item_idx < len(current_page_items):
                        item = current_page_items[item_idx]
                        res_data = item.get('res_data', {})
                        ticker_found = res_data.get('Ticker', '')
                        company_name_found = res_data.get('longNameEn', ticker_found)
                        sector_found = res_data.get('sectorTh', 'N/A')
                        industry_found = res_data.get('industryTh', 'N/A')
                        summary_found = res_data.get('summaryTh', 'N/A')
                        exchange_found = res_data.get('Exchange', 'US Market')
                        raw_df_found = item.get('raw_df')
                        status_lbl = res_data.get('status_text', '')
                        badge_status_class = res_data.get('badge_class', 'badge-board-sideways')
                        pat_found = res_data.get('pattern_name', 'สร้างฐานสะสมกำลัง.png')
                        pat_sc_found = res_data.get('pattern_score', 75.0)

                        with cols[c_offset]:
                            with st.container():
                                st.markdown(f'<p style="font-size:0.95rem; font-weight:bold; color:#60A5FA; margin-bottom:2px;">🟢 {ticker_found} : {company_name_found} <span class="price-badge badge-market">🏛️ {exchange_found}</span></p>', unsafe_allow_html=True)
                                st.markdown(f'<div class="sector-badge" style="font-size:0.72rem; padding:2px 6px; margin-bottom:4px;">🏷️ {sector_found} | ย่อย: {industry_found}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="margin-bottom:6px;"><span class="price-badge {badge_status_class}">{status_lbl}</span></div>', unsafe_allow_html=True)
                                st.caption(f"Support 1: ${res_data.get('Support 1 ($)', 0)} | ต้าน 1: ${res_data.get('Resist 1 ($)', 0)} | RSI: {res_data.get('RSI', 0)}")
                                
                                if raw_df_found is not None:
                                    st.markdown(f'<div class="chart-header-badge">{ticker_found} ({exchange_found}) | ล่าสุด: ${res_data.get("Price ($)", 0)} (RSI: {res_data.get("RSI", 0)})</div>', unsafe_allow_html=True)
                                    fig_gallery = create_ta_chart(raw_df_found, ticker_found, res_data)
                                    if fig_gallery is not None:
                                        st.plotly_chart(fig_gallery, use_container_width=True, config=PLOTLY_CONFIG, key=f"gallery_chart_{ticker_found}_{page_num}_{item_idx}")
                                
                                    st.markdown(f'<div class="pattern-box" style="font-size:0.72rem; padding:3px 6px;">🤖 AI Pattern: {pat_found} ({pat_sc_found}%)</div>', unsafe_allow_html=True)
                                    if summary_found and summary_found != 'N/A':
                                        with st.expander(f"📖 สรุปลักษณะธุรกิจ {ticker_found}", expanded=False):
                                            st.markdown(f'<div class="biz-summary">{summary_found}</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("ไม่พบข้อมูลกราฟ")
                                st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 ตารางสรุปสัญญาณราคาหุ้นทรงสวยประจำวัน")
        if filtered_results:
            all_res_filtered = [item['res_data'] for item in filtered_results]
            df_display_filtered = pd.DataFrame(all_res_filtered)
            cols_to_show = [
                'Ticker', 'longNameEn', 'Exchange', 'sectorTh', 'Price ($)', 'status_text', 
                'pattern_name', 'Support 1 ($)', 'Support 2 ($)', 'Support 3 ($)', 'RSI', 
                'Resist 1 ($)', 'Resist 2 ($)', 'Resist 3 ($)', 'Resist 4 ($)', 
                'Volume', 'Date'
            ]
            available_cols = [c for c in cols_to_show if c in df_display_filtered.columns]
            df_final_show = df_display_filtered[available_cols]

            st.dataframe(df_final_show, use_container_width=True, hide_index=True, height=220)
            st.download_button(
                label='📥 ดาวน์โหลด Watchlist รายการที่กรอง (CSV)',
                data=df_final_show.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                file_name=f'us_watchlist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                key="btn_download_csv"
            )


# --- TAB 4: วิเคราะห์ Forex & ทองคำ (XAUUSD) ---
with tab3:
    st.markdown("### 💱 วิเคราะห์ Forex & สินค้าโภคภัณฑ์ (ทองคำ XAUUSD, น้ำมัน, คริปโต)")
    
    st.markdown("##### ⚡ เลือกคู่เงินยอดนิยม:")
    c_btn1, c_btn2, c_btn3, c_btn4, c_btn5, c_btn6 = st.columns(6)
    if 'selected_forex' not in st.session_state:
        st.session_state.selected_forex = 'XAUUSD'

    if c_btn1.button("🥇 XAUUSD (ทองคำ)"): st.session_state.selected_forex = 'XAUUSD'
    if c_btn2.button("🇪🇺 EURUSD"): st.session_state.selected_forex = 'EURUSD'
    if c_btn3.button("🇬🇧 GBPUSD"): st.session_state.selected_forex = 'GBPUSD'
    if c_btn4.button("🇯🇵 USDJPY"): st.session_state.selected_forex = 'USDJPY'
    if c_btn5.button("₿ BTCUSD"): st.session_state.selected_forex = 'BTCUSD'
    if c_btn6.button("🛢️ USOIL (น้ำมัน)"): st.session_state.selected_forex = 'USOIL'

    col_fx1, col_fx2 = st.columns([3, 1])
    with col_fx1:
        forex_pair = st.text_input(UI_LANG_MAP['search_forex_label'], value=st.session_state.selected_forex).strip().upper()
    with col_fx2:
        st.markdown("<div class='desktop-only-space'></div>", unsafe_allow_html=True)
        forex_search_btn = st.button("🔎 วิเคราะห์คู่เงินทันที", key="btn_search_forex")

    if (forex_search_btn or st.session_state.get('auto_run_fx', False)) and forex_pair:
        with st.spinner(f"⏳ กำลังวิเคราะห์สัญญาณเทคนิคอล {forex_pair}..."):
            res_fx, raw_df_fx = check_ma_snr_combo(forex_pair, info_mode=True)
            if res_fx:
                render_analysis_view(res_fx, raw_df_fx, None, [], forex_pair, is_forex=True)
            else:
                st.error(f"❌ ไม่สามารถดึงข้อมูลสัญลักษณ์ **{forex_pair}** ได้ กรุณาตรวจสอบชื่อคู่เงินอีกครั้ง")

    st.markdown("---")
    st.markdown("#### 🚀 สแกนภาพรวมตลาด Forex, ทองคำ & คริปโต ทั้งหมด")
    
    col_scan_fx1, col_scan_fx2 = st.columns([3, 1])
    with col_scan_fx1:
        btn_run_forex_scan = st.button("🚀 เริ่มสแกนตลาด Forex & สินทรัพย์หลักทั้งหมด", key="btn_run_forex_all")
    with col_scan_fx2:
        btn_reset_fx = st.button("🔄 รีเซ็ตข้อมูล Forex", key="btn_reset_fx")

    if btn_reset_fx:
        server_state["forex_results"] = None
        server_state["forex_df"] = None
        server_state["forex_scanned_at"] = None
        st.success("รีเซ็ตผลการสแกน Forex เรียบร้อยแล้ว")
        st.rerun()

    if btn_run_forex_scan:
        check_ma_snr_combo.clear()
        with st.spinner("⏳ กำลังสแกนคู่เงินหลัก, ทองคำ XAUUSD และสินทรัพย์การเงินทั้งหมด..."):
            fx_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = {executor.submit(check_ma_snr_combo, item, True): item for item in FOREX_DIRECTORY}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res_fx_found, raw_df_fx_found = future.result()
                        if res_fx_found and raw_df_fx_found is not None:
                            fx_results.append({'res_data': res_fx_found, 'raw_df': raw_df_fx_found})
                    except Exception:
                        pass

            server_state["forex_results"] = fx_results
            server_state["forex_scanned_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if fx_results:
                all_fx_data = [item['res_data'] for item in fx_results]
                df_fx_display = pd.DataFrame(all_fx_data)
                cols_to_show_fx = [
                    'Ticker', 'longNameEn', 'Exchange', 'Price ($)', 'status_text', 
                    'pattern_name', 'Support 1 ($)', 'Support 2 ($)', 'Support 3 ($)', 'RSI', 
                    'Resist 1 ($)', 'Resist 2 ($)', 'Resist 3 ($)', 'Resist 4 ($)', 'Date'
                ]
                available_fx_cols = [c for c in cols_to_show_fx if c in df_fx_display.columns]
                server_state["forex_df"] = df_fx_display[available_fx_cols]
            st.rerun()

    if server_state["forex_results"]:
        fx_results = server_state["forex_results"]
        st.info(f"🕒 สแกนล่าสุดเมื่อ: **{server_state.get('forex_scanned_at', '')}** | พบสินทรัพย์ทั้งหมด **{len(fx_results)}** รายการ")
        
        for row_idx in range(0, len(fx_results), 2):
            cols = st.columns(2)
            for c_offset in range(2):
                item_idx = row_idx + c_offset
                if item_idx < len(fx_results):
                    item = fx_results[item_idx]
                    res_data = item.get('res_data', {})
                    ticker_found = res_data.get('Ticker', '')
                    company_name_found = res_data.get('longNameEn', ticker_found)
                    sector_found = res_data.get('sectorTh', 'Forex/Commodity')
                    exchange_found = res_data.get('Exchange', 'Global Market')
                    raw_df_found = item.get('raw_df')
                    status_lbl = res_data.get('status_text', '')
                    badge_status_class = res_data.get('badge_class', 'badge-board-sideways')
                    pat_found = res_data.get('pattern_name', 'สร้างฐานสะสมกำลัง.png')
                    pat_sc_found = res_data.get('pattern_score', 75.0)

                    with cols[c_offset]:
                        with st.container():
                            st.markdown(f'<p style="font-size:0.95rem; font-weight:bold; color:#60A5FA; margin-bottom:2px;">🟢 {ticker_found} : {company_name_found} <span class="price-badge badge-market">🏛️ {exchange_found}</span></p>', unsafe_allow_html=True)
                            st.markdown(f'<div class="sector-badge" style="font-size:0.72rem; padding:2px 6px; margin-bottom:4px;">🏷️ {sector_found}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="margin-bottom:6px;"><span class="price-badge {badge_status_class}">{status_lbl}</span></div>', unsafe_allow_html=True)
                            st.caption(f"Support 1: {res_data.get('Support 1 ($)', 0)} | ต้าน 1: {res_data.get('Resist 1 ($)', 0)} | RSI: {res_data.get('RSI', 0)}")
                            
                            if raw_df_found is not None:
                                st.markdown(f'<div class="chart-header-badge">{ticker_found} ({exchange_found}) | ล่าสุด: {res_data.get("Price ($)", 0)} (RSI: {res_data.get("RSI", 0)})</div>', unsafe_allow_html=True)
                                fig_gallery_fx = create_ta_chart(raw_df_found, ticker_found, res_data)
                                if fig_gallery_fx is not None:
                                    st.plotly_chart(fig_gallery_fx, use_container_width=True, config=PLOTLY_CONFIG, key=f"gal_fx_{ticker_found}_{item_idx}")
                            
                                st.markdown(f'<div class="pattern-box" style="font-size:0.72rem; padding:3px 6px;">🤖 AI Pattern: {pat_found} ({pat_sc_found}%)</div>', unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

        if server_state.get("forex_df") is not None:
            st.markdown("---")
            st.markdown("#### 📊 ตารางสรุปสัญญาณราคา Forex & ทองคำ")
            st.dataframe(server_state["forex_df"], use_container_width=True, hide_index=True, height=220)


# --- TAB 5: ข่าวเด่นเศรษฐกิจ & ปัจจัยตลาดหุ้น (Macro Market News) ---
with tab_news:
    st.markdown("### 📰 ข่าวเด่นเศรษฐกิจมหภาค & ปัจจัยกระทบตลาดหุ้น (Macro News)")
    st.caption("อัปเดตข่าวสารทิศทางดอกเบี้ย Fed, เศรษฐกิจสหรัฐฯ และความเคลื่อนไหวตลาดการเงินโลก (คลิกเพื่ออ่านข่าวต้นฉบับ)")
    
    if st.button("🔄 อัปเดตข่าวสารล่าสุด", key="btn_refresh_macro_news"):
        get_macro_market_news.clear()
        st.rerun()

    with st.spinner("⏳ กำลังดึงและแปลหัวข้อข่าวเศรษฐกิจล่าสุด..."):
        macro_news_list = get_macro_market_news()

    if macro_news_list:
        for news in macro_news_list:
            pub_time_str = f" • เผยแพร่เมื่อ: {news['time']}" if news['time'] else ""
            st.markdown(f"""
            <a href="{news['link']}" target="_blank" class="news-card-link">
                <div class="news-card-title">📌 {news['title']}</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:3px; font-style:italic;">En: {news['title_en']}</div>
                <div class="news-card-meta">🌐 แหล่งข่าว: Financial Market News{pub_time_str}</div>
            </a>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ ขณะนี้ไม่สามารถเชื่อมต่อฟีดข่าวเศรษฐกิจได้ กรุณากดปุ่มอัปเดตอีกครั้ง")
