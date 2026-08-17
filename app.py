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

# ================= 1. ตั้งค่าแอปและตัวแปรหลัก =================
st.set_page_config(
    page_title="US Stock Scanner PRO (by.Jetsada)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ตัวแปรตั้งค่า Plotly Toolbar
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True
}

# พจนานุกรมข้อความและภาษา
UI_LANG_MAP = {
    'search_ticker_title': "US Stock Scanner PRO (by.Jetsada)",
    'search_ticker_subtitle': "ระบบสแกนเทคนิคอล • คำนวณ % โครงสร้างราคา • AI Pattern • 3 แนวรับ 4 แนวต้าน",
    'search_ticker_label': "พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, RKLB, AAOI, RXT, CRWV, BZAI):",
    'btn_analyze_single': "🔎 วิเคราะห์ทันที",
    'btn_scan_market': "🚀 เริ่มสแกนตลาด",
    'status_preparing_tickers': "⏳ กำลังดึงรายชื่อหุ้นผู้นำตลาด (S&P 500, NASDAQ 100, Growth)...",
    'status_scanning': "⏳ สแกนไปแล้ว {count}/{total} ตัว (พบหุ้นทรงสวย {found} ตัว)...",
    'status_analyzing_single': "⏳ กำลังดึงข้อมูลสดและวิเคราะห์ {ticker}...",
    'expander_business_summary': "📖 สรุปธุรกิจ & โครงสร้างผู้ถือหุ้น (แปลไทยอัตโนมัติ)",
    'chart_title_single': "📈 กราฟเทคนิค 3 แนวรับ และ 4 ระดับแนวต้าน",
    'analysis_title': "📊 ข้อมูลแนวรับ - แนวต้าน & ตัวชี้วัดสำคัญ",
    'tab_search_ticker': "🔍 ค้นหา & วิเคราะห์รายตัว",
    'tab_scan_market': "🚀 สแกนคัดหุ้นทรงสวย",
    'tab_watchlist': "⭐ Watchlist ส่วนตัว",
}

SECTOR_MAP_TH = {
    'Technology': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์',
    'Healthcare': '🏥 สุขภาพ / การแพทย์ & ยา',
    'Financial Services': '🏦 การเงิน / ธนาคาร & ประกันภัย',
    'Industrials': '🏭 อุตสาหกรรม / อวกาศ & ขนส่ง',
    'Consumer Cyclical': '🛍️ สินค้าฟุ่มเฟือย / ค้าปลีก & ยานยนต์',
    'Consumer Defensive': '🛒 สินค้าอุปโภคบริโภคจำเป็น',
    'Energy': '⚡ พลังงาน / น้ำมัน & ก๊าซ',
    'Real Estate': '🏢 อสังหาริมทรัพย์ / กองรีท (REITs)',
    'Basic Materials': '🧪 วัตถุดิบพื้นฐาน / เคมีภัณฑ์ & เหมืองแร่',
    'Communication Services': '📡 สื่อสาร / โทรคมนาคม & บันเทิง',
    'Utilities': '💡 สาธารณูปโภค / ไฟฟ้า & ประปา'
}

# ================= 2. จัดการ Session และ Global State =================
@st.cache_resource
def get_yfinance_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
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
        "last_scanned_dt": None
    }

server_state = get_global_server_state()

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# ================= 3. Custom CSS ปรับแต่งสีพื้นหลัง 5 สีตามสถานะ =================
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
    
    /* กล่องสถานะตามสี 5 สภาวะตลาด */
    .status-banner {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 0.8rem;
        line-height: 1.5;
        box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    }
    .status-banner-uptrend {
        background-color: #022c22 !important;
        border: 1.5px solid #10b981 !important;
        color: #a7f3d0 !important;
    }
    .status-banner-pullback {
        background-color: #451a03 !important;
        border: 1.5px solid #f59e0b !important;
        color: #fef08a !important;
    }
    .status-banner-support {
        background-color: #172554 !important;
        border: 1.5px solid #38bdf8 !important;
        color: #bae6fd !important;
    }
    .status-banner-sideways {
        background-color: #1e293b !important;
        border: 1.5px solid #94a3b8 !important;
        color: #f1f5f9 !important;
    }
    .status-banner-downtrend {
        background-color: #4c0519 !important;
        border: 1.5px solid #f43f5e !important;
        color: #fecdd3 !important;
    }
    .status-title-text {
        font-size: 1.0rem;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -0.2px;
    }
    .status-desc-text {
        font-size: 0.84rem;
        opacity: 0.95;
    }

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
    .price-val-box {
        display: flex;
        align-items: baseline;
        gap: 6px;
    }
    .price-main {
        font-size: 1.45rem;
        font-weight: 900;
        color: #F8FAFC;
    }
    .price-badge-group {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    .price-badge {
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 5px;
        font-weight: 600;
        white-space: nowrap;
    }
    .badge-rsi { background: #1E293B; color: #38BDF8; border: 1px solid #334155; }
    .badge-dist { background: #064E3B; color: #34D399; border: 1px solid #059669; }
    .badge-trend-bull { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
    .badge-trend-bear { background: #4C0519; color: #FDA4AF; border: 1px solid #9F1239; }
    .badge-trend-pull { background: #451A03; color: #FCD34D; border: 1px solid #78350F; }
    .badge-trend-support { background: #1E3A8A; color: #93C5FD; border: 1px solid #1D4ED8; }
    .badge-trend-side { background: #1E293B; color: #94A3B8; border: 1px solid #475569; }
    .badge-ai-box { background: #172554; color: #93C5FD; border: 1px solid #1E40AF; font-weight: 700; }

    .snr-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .snr-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 6px 8px;
    }
    .snr-card-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 4px;
        padding-bottom: 3px;
        border-bottom: 1px dashed #334155;
    }
    .snr-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        padding: 2px 0;
    }
    .snr-lbl { color: #94A3B8; font-size: 0.72rem; }
    .snr-num { font-weight: 700; font-size: 0.82rem; }
    
    .c-green { color: #22C55E !important; }
    .c-lightgreen { color: #4ADE80 !important; }
    .c-red { color: #EF4444 !important; }
    .c-orange { color: #F97316 !important; }
    .c-yellow { color: #FBBF24 !important; }
    .c-darkred { color: #F43F5E !important; }

    .strategy-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .strat-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .strat-title { font-size: 0.82rem; font-weight: 700; color: #F8FAFC; }
    .strat-price { font-size: 0.95rem; font-weight: 800; color: #38BDF8; }
    .strat-body {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.74rem;
        padding-top: 4px;
        border-top: 1px dashed #1E293B;
        flex-wrap: wrap;
        gap: 4px;
    }
    .strat-sub { color: #94A3B8; }
    .strat-val { color: #F8FAFC; font-weight: 600; }

    .company-header { font-size: 1.15rem; font-weight: 800; color: #38BDF8 !important; margin-bottom: 0rem; }
    .sector-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: #FCD34D;
        background: #451A03;
        border: 1px solid #78350F;
        padding: 3px 7px;
        border-radius: 5px;
        display: inline-block;
        margin-top: 3px;
        margin-bottom: 4px;
    }
    .chart-header-badge {
        font-size: 0.82rem;
        font-weight: 700;
        color: #F8FAFC;
        background-color: #1E293B;
        padding: 4px 7px;
        border-radius: 5px;
        margin-bottom: 3px;
        display: inline-block;
    }
    .fin-card {
        background: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 0.4rem;
        color: #F8FAFC !important;
    }
    .biz-summary {
        font-size: 0.82rem !important;
        color: #F1F5F9 !important;
        background-color: #0B132B !important;
        padding: 10px !important;
        border-radius: 6px;
        border-left: 3px solid #3B82F6 !important;
        border: 1px solid #334155 !important;
        margin-bottom: 0.3rem;
        line-height: 1.5;
    }
    .pattern-box {
        background-color: #172554 !important;
        color: #93C5FD !important;
        padding: 5px 8px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid #1E40AF !important;
        margin-top: 3px;
        margin-bottom: 4px;
    }
    .news-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .news-title {
        font-size: 0.84rem;
        font-weight: 600;
        color: #60A5FA !important;
        text-decoration: none;
        display: block;
        margin-bottom: 3px;
    }
    .news-title:hover { text-decoration: underline; }
    .news-meta { font-size: 0.72rem; color: #94A3B8; }
    
    .desktop-only-space { height: 28px; display: block; }
    @media (max-width: 640px) {
        .desktop-only-space { display: none !important; }
        .main-title { font-size: 1.3rem !important; }
        .price-main { font-size: 1.3rem; }
        .snr-row { font-size: 0.72rem; }
        .snr-num { font-size: 0.78rem; }
        .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="main-title">{UI_LANG_MAP["search_ticker_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{UI_LANG_MAP["search_ticker_subtitle"]}</div>', unsafe_allow_html=True)

# ================= 4. ระบบดึงรายชื่อหุ้น S&P 500, NASDAQ 100 และ Growth ตัวจริง =================
@st.cache_data(ttl=86400)
def get_us_stock_tickers(scope="TOP500"):
    tickers = []
    
    # 1. ดึง S&P 500 ตัวจริงจาก Wikipedia (ครบทุกหมวดตัวอักษร A-Z)
    try:
        url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url_sp500)
        sp500_df = tables[0]
        sp500_tickers = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()
        tickers.extend(sp500_tickers)
    except Exception:
        pass

    # 2. ดึง NASDAQ 100 ตัวจริง
    try:
        url_nasdaq = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables_nd = pd.read_html(url_nasdaq)
        # ตารางรายชื่อ NASDAQ 100 มักเป็นตารางที่ 4 หรือ 5
        for t in tables_nd:
            if 'Ticker' in t.columns:
                tickers.extend(t['Ticker'].str.replace('.', '-', regex=False).tolist())
                break
            elif 'Symbol' in t.columns:
                tickers.extend(t['Symbol'].str.replace('.', '-', regex=False).tolist())
                break
    except Exception:
        pass

    # 3. หุ้น Growth & Momentum ยอดนิยมที่ต้องมีเสมอ
    top_growth_stocks = [
        'NVDA', 'PLTR', 'TSLA', 'AMD', 'ARM', 'SMCI', 'RKLB', 'AAOI', 'CRWV', 'RXT', 'BZAI',
        'SOFI', 'MARA', 'RIOT', 'COIN', 'HOOD', 'MSTR', 'DKNG', 'HIMS', 'APP', 'ASTS', 'RDDT',
        'AFRM', 'IREN', 'WULF', 'CIFR', 'CLSK', 'IONQ', 'RGTI', 'QBTS', 'SOUN', 'BBAI', 'AI',
        'PATH', 'SNOW', 'MDB', 'DDOG', 'ZS', 'NET', 'CRWD', 'PANW', 'FTNT', 'OKTA', 'AVGO',
        'MRVL', 'ON', 'MPWR', 'ALAB', 'VRT', 'POWI', 'AMAT', 'LRCX', 'KLAC', 'ASML', 'TSM'
    ]
    tickers.extend(top_growth_stocks)

    # 4. หากเลือกสแกนทั้งหมด (ALL) ดึงจาก SEC API เพิ่มเติม
    if scope == "ALL":
        try:
            sec_url = "https://www.sec.gov/files/company_tickers.json"
            headers = {'User-Agent': 'USStockScannerApp/2.0 (admin@stockscannerpro.org)'}
            r = requests.get(sec_url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                for item in data.values():
                    t = str(item.get('ticker', '')).strip().upper().replace('.', '-')
                    if t and t.isalpha() and len(t) <= 5:
                        tickers.append(t)
        except Exception:
            pass

    # กรอง Ticker ซ้ำและทำความสะอาด
    clean_tickers = []
    seen = set()
    for t in tickers:
        t_clean = str(t).strip().upper()
        if t_clean and t_clean not in seen and len(t_clean) <= 6:
            clean_tickers.append(t_clean)
            seen.add(t_clean)

    if scope == "TOP500":
        return clean_tickers[:500]
    elif scope == "GROWTH1000":
        return clean_tickers[:1000]
    return clean_tickers[:2500] # สแกนชุดสภาพคล่องสูง 2,500 ตัวเพื่อความเร็วและไม่โดนบล็อก

# ================= 5. ฟังก์ชันดึงประวัติราคา Dual-Engine =================
def fetch_stock_history_dual(ticker):
    ticker_clean = str(ticker).strip().upper()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }

    # 1. Direct Yahoo API v8
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_clean}?range=6mo&interval=1d"
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            data = res.json()
            result = data.get('chart', {}).get('result', [])
            if result:
                r = result[0]
                timestamps = r.get('timestamp', [])
                quote = r.get('indicators', {}).get('quote', [{}])[0]
                if timestamps and quote:
                    df = pd.DataFrame({
                        'open': quote.get('open', []),
                        'high': quote.get('high', []),
                        'low': quote.get('low', []),
                        'close': quote.get('close', []),
                        'volume': quote.get('volume', [])
                    }, index=pd.to_datetime(timestamps, unit='s'))
                    df = df.dropna(subset=['close'])
                    if len(df) >= 15:
                        return df
    except Exception:
        pass

    # 2. สำรองด้วย yfinance
    try:
        stock = yf.Ticker(ticker_clean)
        df = stock.history(period='6mo', interval='1d')
        if df is not None and not df.empty and len(df) >= 15:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            return df
    except Exception:
        pass

    return None


def calculate_swing_snr(df, latest_close):
    n = len(df)
    window_n = min(n, 120)
    df_wave = df.iloc[-window_n:]
    
    highs = df_wave['high'].values
    lows = df_wave['low'].values
    
    wave_high = float(np.max(highs))
    wave_low = float(np.min(lows))
    wave_range = max(1e-4, wave_high - wave_low)

    fib_236 = wave_high - 0.236 * wave_range
    fib_382 = wave_high - 0.382 * wave_range
    fib_500 = wave_high - 0.500 * wave_range
    fib_618 = wave_high - 0.618 * wave_range

    recent_15d_low = float(np.min(lows[-15:]))
    recent_45d_low = float(np.min(lows[-45:]))

    # กำหนดแนวรับ
    if recent_15d_low < latest_close * 0.995 and recent_15d_low > latest_close * 0.85:
        s1 = recent_15d_low
    elif fib_500 < latest_close * 0.995 and fib_500 > latest_close * 0.88:
        s1 = fib_500
    elif fib_618 < latest_close * 0.995:
        s1 = fib_618
    else:
        s1 = latest_close * 0.95

    if recent_45d_low < s1 * 0.96 and recent_45d_low > wave_low * 1.15:
        s2 = recent_45d_low
    elif fib_618 < s1 * 0.96:
        s2 = fib_618
    else:
        s2 = s1 * 0.89

    s3 = wave_low if wave_low < s2 * 0.90 else s2 * 0.75

    if s2 >= s1: s2 = s1 * 0.90
    if s3 >= s2: s3 = s2 * 0.75

    # กำหนดแนวต้าน
    r4 = wave_high
    cand_resists = [fib_500, fib_382, fib_236]
    valid_resists = sorted([r for r in cand_resists if r > latest_close * 1.015 and r < r4 * 0.985])

    if len(valid_resists) >= 3:
        r1, r2, r3 = valid_resists[0], valid_resists[1], valid_resists[2]
    elif len(valid_resists) == 2:
        r1, r2 = valid_resists[0], valid_resists[1]
        r3 = r2 + (r4 - r2) * 0.50
    elif len(valid_resists) == 1:
        r1 = valid_resists[0]
        r2 = r1 + (r4 - r1) * 0.35
        r3 = r1 + (r4 - r1) * 0.70
    else:
        r1 = latest_close * 1.06
        r2 = latest_close * 1.15
        r3 = latest_close * 1.25

    return round(s1, 2), round(s2, 2), round(s3, 2), round(r1, 2), round(r2, 2), round(r3, 2), round(r4, 2)


def calculate_ai_pattern_match(df):
    try:
        if df is None or len(df) < 15:
            return "สร้างฐานสะสมกำลัง.png", 75.0

        bars = min(len(df), 25)
        closes = df['close'].tail(bars).values
        c_min, c_max = np.min(closes), np.max(closes)
        if c_max == c_min:
            return "สร้างฐานสะสมกำลัง.png", 82.0
        norm_closes = (closes - c_min) / (c_max - c_min)
        
        x = np.linspace(0, 1, bars)
        templates = {
            "สร้างฐานยก Low.png": 0.15 + 0.75 * x + 0.08 * np.sin(x * 3 * np.pi),
            "สร้างฐานแบบ Double Bottom.png": 0.65 - 0.65 * np.sin(x * np.pi) + 0.25 * np.cos(x * 2 * np.pi),
            "สร้างฐานก้นกระทะ (Rounding).png": 0.85 - 0.85 * np.sin(x * np.pi),
            "สร้างฐานสะสมกำลัง.png": np.full(bars, 0.5) + 0.08 * np.sin(x * 5 * np.pi),
            "ทรงหลุดฐานขาลง.png": 0.9 - 0.8 * x + 0.05 * np.sin(x * 4 * np.pi)
        }

        best_pattern = "สร้างฐานสะสมกำลัง.png"
        best_score = 60.0

        for pat_name, pat_curve in templates.items():
            norm_pat = (pat_curve - np.min(pat_curve)) / (np.max(pat_curve) - np.min(pat_curve) + 1e-6)
            mae = np.mean(np.abs(norm_closes - norm_pat))
            corr = np.corrcoef(norm_closes, norm_pat)[0, 1]
            if np.isnan(corr): corr = 0.5
            
            sim_score = (max(0.0, 1.0 - mae) * 0.65 + max(0.0, (corr + 1.0) / 2.0) * 0.35) * 100.0
            if sim_score > best_score:
                best_score = sim_score
                best_pattern = pat_name

        return best_pattern, round(max(70.0, min(95.5, best_score)), 1)
    except Exception:
        return "สร้างฐานสะสมกำลัง.png", 76.5


def get_time_elapsed_thai(last_dt):
    if not last_dt: return ""
    diff = datetime.now() - last_dt
    secs = int(diff.total_seconds())
    if secs < 60: return f" (เพิ่งสแกนเมื่อ {secs} วิที่แล้ว)"
    elif secs < 3600: return f" (สแกนไปแล้ว {secs // 60} นาทีที่แล้ว)"
    else: return f" (สแกนไปแล้ว {secs // 3600} ชม. ก่อน)"


def translate_text_to_thai(text):
    if not text or text == 'N/A' or not str(text).strip(): return ''
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "en", "tl": "th", "dt": "t", "q": text}
        res = requests.get(url, params=params, timeout=3)
        if res.status_code == 200:
            return "".join([item[0] for item in res.json()[0] if item[0]])
    except Exception: pass
    return str(text)


@st.cache_data(ttl=1800)
def get_company_info_and_holders(ticker):
    try:
        info = yf.Ticker(ticker, session=get_yfinance_session()).info
        return {
            'longNameEn': info.get('longName', ticker),
            'sectorTh': SECTOR_MAP_TH.get(info.get('sector', ''), info.get('sector', 'N/A')),
            'industryTh': translate_text_to_thai(info.get('industry', 'N/A')),
            'summaryTh': translate_text_to_thai(info.get('longBusinessSummary', 'N/A')),
            'sharesOutstanding': f"{info.get('sharesOutstanding', 0):,.0f}" if info.get('sharesOutstanding') else "N/A",
            'institutionalHeld': f"{info.get('heldPercentInstitutions', 0)*100:.2f}%" if info.get('heldPercentInstitutions') else "N/A",
            'insiderHeld': f"{info.get('heldPercentInsiders', 0)*100:.2f}%" if info.get('heldPercentInsiders') else "N/A",
            'retailHeld': f"{100 - (info.get('heldPercentInstitutions',0)+info.get('heldPercentInsiders',0))*100:.2f}%" if info.get('heldPercentInstitutions') else "N/A"
        }
    except Exception:
        return {'longNameEn': ticker, 'sectorTh': 'N/A', 'industryTh': 'N/A', 'summaryTh': 'N/A', 'sharesOutstanding': 'N/A', 'institutionalHeld': 'N/A', 'insiderHeld': 'N/A', 'retailHeld': 'N/A'}


@st.cache_data(ttl=900)
def get_stock_news(ticker):
    results = []
    try:
        stock = yf.Ticker(ticker, session=get_yfinance_session())
        news_items = stock.news
        if news_items:
            for n in news_items:
                title_en, link, publisher, pub_date_str = "", "#", "Yahoo Finance", ""
                if 'content' in n and isinstance(n['content'], dict):
                    c = n['content']
                    title_en = c.get('title', '')
                    publisher = c.get('provider', {}).get('displayName', 'Financial News')
                    link = c.get('canonicalUrl', {}).get('url', c.get('clickThroughUrl', {}).get('url', '#'))
                    pub_date_str = c.get('pubDate', '')[:16].replace('T', ' ')
                if not title_en:
                    title_en = n.get('title', '')
                    publisher = n.get('publisher', 'Financial News')
                    link = n.get('link', '#')
                    pub_ts = n.get('providerPublishTime', 0)
                    if pub_ts: pub_date_str = datetime.fromtimestamp(pub_ts).strftime('%d/%m/%Y %H:%M')

                if title_en and title_en.strip():
                    title_th = translate_text_to_thai(title_en)
                    results.append({'title': title_th if title_th else title_en, 'publisher': publisher, 'link': link, 'time': pub_date_str})
                if len(results) >= 3: break
    except Exception: pass

    if not results:
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            res = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=4)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall('./channel/item')[:3]:
                    t_node, l_node, p_node = item.find('title'), item.find('link'), item.find('pubDate')
                    raw_title = t_node.text if t_node is not None else ""
                    raw_link = l_node.text if l_node is not None else "#"
                    raw_pub = p_node.text[:16] if p_node is not None else ""
                    if raw_title:
                        title_th = translate_text_to_thai(raw_title)
                        results.append({'title': title_th if title_th else raw_title, 'publisher': 'Yahoo Feed', 'link': raw_link, 'time': raw_pub})
        except Exception: pass

    return results


@st.cache_data(ttl=14400)
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker, session=get_yfinance_session()).quarterly_financials
        if q_financials is not None and 'Net Income' in q_financials.index:
            net_income = q_financials.loc['Net Income'].head(3)
            data = []
            for date, value in net_income.items():
                if pd.notna(value):
                    data.append({
                        'Quarter End': date.strftime('%Y-%m-%d'),
                        'Net Income (M$)': round(value / 1_000_000, 2)
                    })
            if data:
                return pd.DataFrame(data)
    except Exception: pass
    return None


def create_ta_chart(df, ticker, res_data):
    if df is None or df.empty: return None
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='ราคา'
    )])
    fast_ma = df['close'].rolling(20).mean()
    slow_ma = df['close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=fast_ma, line=dict(color='#38BDF8', width=1.2), name='MA20'))
    fig.add_trace(go.Scatter(x=df.index, y=slow_ma, line=dict(color='#FB923C', width=1.2), name='MA50'))

    for key, color, ay_pos in [('Support 1 ($)', '#22C55E', -12), ('Support 2 ($)', '#16A34A', 12), ('Support 3 ($)', '#15803D', -12)]:
        if key in res_data:
            val = res_data[key]
            fig.add_shape(type="line", x0=df.index[0], y0=val, x1=df.index[-1], y1=val, line=dict(color=color, width=1.6, dash='dash'))
            fig.add_annotation(x=df.index[-1], y=val, text=f"{key.replace(' ($)', '')}: ${val}", bgcolor=color, font=dict(color="white", size=9), xanchor="left", ax=8, ay=ay_pos)

    for key, color, ay_pos in [('Resist 1 ($)', '#EF4444', -12), ('Resist 2 ($)', '#F97316', 12), ('Resist 3 ($)', '#EAB308', -12), ('Resist 4 ($)', '#991B1B', 12)]:
        if key in res_data:
            val = res_data[key]
            fig.add_shape(type="line", x0=df.index[0], y0=val, x1=df.index[-1], y1=val, line=dict(color=color, width=1.6, dash='dash'))
            fig.add_annotation(x=df.index[-1], y=val, text=f"{key.replace(' ($)', '')}: ${val}", bgcolor=color, font=dict(color="white", size=9), xanchor="left", ax=8, ay=ay_pos)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', margin=dict(l=6, r=65, t=10, b=6), height=340, dragmode='pan', yaxis_title="ราคา ($)", showlegend=False)
    return fig


# ================= 6. ฟังก์ชันวิเคราะห์หลัก (จำแนกลักษณะแท่งเทียน 1D อย่างแม่นยำ) =================
@st.cache_data(ttl=300)
def check_ma_snr_combo(ticker, info_mode=False):
    try:
        df = fetch_stock_history_dual(ticker)
        if df is None or df.empty or len(df) < 15:
            return None, None

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

        s1, s2, s3, r1, r2, r3, r4 = calculate_swing_snr(df, latest_close)

        recent_8d_high = float(df['high'].tail(8).max())
        drop_8d_pct = ((latest_close - recent_8d_high) / recent_8d_high) * 100
        recent_8d_low = float(df['low'].tail(8).min())
        bounce_8d_pct = ((latest_close - recent_8d_low) / recent_8d_low) * 100 if recent_8d_low > 0 else 0.0

        is_above_ma20 = latest_close >= (fast_ma.iloc[-1] if pd.notna(fast_ma.iloc[-1]) else latest_close)
        is_above_ma50 = latest_close >= (slow_ma.iloc[-1] if pd.notna(slow_ma.iloc[-1]) else latest_close)
        is_ma_bull = (fast_ma.iloc[-1] >= slow_ma.iloc[-1]) if (pd.notna(fast_ma.iloc[-1]) and pd.notna(slow_ma.iloc[-1])) else False

        # คำนวณคะแนนภาพรวม
        bull_score = 0
        if is_above_ma20: bull_score += 30
        if is_above_ma50: bull_score += 25
        if is_ma_bull: bull_score += 20
        if 50 <= latest_rsi <= 72: bull_score += 15
        elif 40 <= latest_rsi < 50: bull_score += 5
        if drop_8d_pct > -8: bull_score += 10
        bullish_pct = min(96.0, max(5.0, round(bull_score * 0.95 + 4.0, 1)))

        dist_s1_pct = ((latest_close - s1) / s1) * 100

        # ================= ตรรกะ 5 สภาวะตลาดแท้จริง (ตรงตามพฤติกรรมแท่งเทียน 1D) =================
        # 1. ขาลงชัดเจน / โดนทุบหนัก
        if (not is_above_ma20 and not is_above_ma50 and latest_rsi < 45) or drop_8d_pct <= -15.0:
            trend_status = "DOWNTREND"
            status_text = "📉 ลงแรง / ขาลงชัดเจน (ห้ามรับมีด)"
            status_box_class = "status-banner-downtrend"
            badge_class = "badge-trend-bear"
            badge_label = f"📉 ลงแรง/ขาลง: {100.0 - bullish_pct:.1f}%"
            status_desc = f"⚠️ หุ้นหลุดเส้นค่าเฉลี่ยหลัก (ย่อตัวจากยอด 8 วัน {drop_8d_pct:.2f}%) โครงสร้างเสียเปรียบ ยังไม่ควรรับมีด"

        # 2. ย่อพักฐาน (Healthy Pullback)
        elif (is_above_ma50 or is_ma_bull or bounce_8d_pct >= 8.0) and (drop_8d_pct <= -3.0 or not is_today_green) and latest_rsi >= 40:
            trend_status = "PULLBACK"
            status_text = "⏳ ย่อพักฐาน (Healthy Pullback)"
            status_box_class = "status-banner-pullback"
            badge_class = "badge-trend-pull"
            badge_label = f"⏳ ย่อพักฐาน: {bullish_pct}%"
            status_desc = f"🔄 หุ้นอยู่ในแนวโน้มใหญ่ขาขึ้น แต่แท่งเทียนกำลังย่อตัวพักฐานตามรอบ (ย่อจากยอด 8 วัน {drop_8d_pct:.2f}%) เพื่อสะสมแรง"

        # 3. ช้อนแนวรับ
        elif dist_s1_pct <= 4.5 and bounce_8d_pct <= 5.5 and latest_close >= s1 * 0.98:
            trend_status = "BUY_SUPPORT"
            status_text = f"🎯 ช้อนแนวรับ (เด้งจากฐาน +{bounce_8d_pct:.2f}%)"
            status_box_class = "status-banner-support"
            badge_class = "badge-trend-support"
            badge_label = f"🎯 ช้อนแนวรับ: {bullish_pct}%"
            status_desc = f"🛡️ ราคาอยู่ในโซนแนวรับสำคัญและเริ่มมีแรงดีดกลับตัว (+{bounce_8d_pct:.2f}%) เหมาะสะสมไม้ 1"

        # 4. ขาขึ้นแข็งแกร่ง
        elif is_above_ma20 and is_above_ma50 and latest_rsi >= 50 and drop_8d_pct > -3.0:
            trend_status = "UPTREND"
            status_text = "🚀 ขาขึ้นแข็งแกร่ง (Strong Uptrend)"
            status_box_class = "status-banner-uptrend"
            badge_class = "badge-trend-bull"
            badge_label = f"🚀 ขาขึ้นแข็งแกร่ง: {bullish_pct}%"
            status_desc = f"✨ ราคายืนเหนือเส้นแนวโน้มหลักทุกเส้น โมเมนตัมขาขึ้นสมบูรณ์ (ดีดตัวจากฐานล่าสุด +{bounce_8d_pct:.2f}%)"

        # 5. สะสมแรง / ไซด์เวย์
        else:
            trend_status = "SIDEWAYS"
            status_text = "〰️ สะสมแรง / ไซด์เวย์"
            status_box_class = "status-banner-sideways"
            badge_class = "badge-trend-side"
            badge_label = f"〰️ สะสมแรง/ไซด์เวย์: {bullish_pct}%"
            status_desc = f"📦 ราคาแกว่งตัวสร้างฐานในกรอบแคบ ยังไม่มีทิศทางชัดเจน รอการเบรกเอาท์"

        dist_from_sup = ((latest_close - s1) / s1) * 100
        pat_name, pat_score = calculate_ai_pattern_match(df.tail(60))
        vol_val = df['volume'].iloc[-1] if 'volume' in df.columns else 0

        res_data = {
            'Ticker': ticker,
            'Price ($)': round(latest_close, 2),
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

        if info_mode:
            res_data.update(get_company_info_and_holders(ticker))

        if not info_mode:
            if not (trend_status in ["UPTREND", "PULLBACK", "BUY_SUPPORT"]):
                return None, df

        return res_data, df
    except Exception:
        pass
    return None, None


# ================= 7. ส่วนแสดงผล UI หน้าจอ =================
tab1, tab2, tab3 = st.tabs([UI_LANG_MAP['tab_search_ticker'], UI_LANG_MAP['tab_scan_market'], UI_LANG_MAP['tab_watchlist']])

# --- TAB 1: ค้นหาหุ้นรายตัว ---
with tab1:
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        single_ticker = st.text_input(UI_LANG_MAP['search_ticker_label'], value='CRWV').strip().upper()
    with col_in2:
        st.markdown("<div class='desktop-only-space'></div>", unsafe_allow_html=True)
        search_btn = st.button(UI_LANG_MAP['btn_analyze_single'])

    if search_btn and single_ticker:
        with st.spinner(UI_LANG_MAP['status_analyzing_single'].format(ticker=single_ticker)):
            res, raw_df = check_ma_snr_combo(single_ticker, info_mode=True)
            df_profit = get_financials(single_ticker)
            news_items = get_stock_news(single_ticker)

            if res:
                company_full_name = res.get("longNameEn", single_ticker)
                sector_desc = res.get("sectorTh", "N/A")
                industry_desc = res.get("industryTh", "N/A")

                st.markdown(f'<p class="company-header">{single_ticker} : {company_full_name}</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="sector-badge">🏷️ กลุ่มธุรกิจ: {sector_desc} | ย่อย: {industry_desc}</div>', unsafe_allow_html=True)
                
                # แสดงผลแถบสถานะพร้อมเปลี่ยนสีพื้นหลัง 5 สีอย่างชัดเจน
                box_css = res.get('status_box_class', 'status-banner-sideways')
                st.markdown(f"""
                <div class="status-banner {box_css}">
                    <div class="status-title-text">{res.get('status_text', '')}</div>
                    <div class="status-desc-text">{res.get('status_desc', '')} | ข้อมูล ณ วันที่: {res.get('Date', '')}</div>
                </div>
                """, unsafe_allow_html=True)

                # ปุ่ม Watchlist
                if single_ticker not in st.session_state.watchlist:
                    if st.button(f"⭐ เพิ่ม {single_ticker} เข้า Watchlist", key=f"btn_add_wl_{single_ticker}"):
                        st.session_state.watchlist.append(single_ticker)
                        st.success(f"เพิ่ม {single_ticker} สำเร็จ!")
                        st.rerun()
                else:
                    if st.button(f"🗑️ ลบ {single_ticker} ออกจาก Watchlist", key=f"btn_del_wl_{single_ticker}"):
                        st.session_state.watchlist.remove(single_ticker)
                        st.rerun()
                    st.info(f"📌 หุ้น {single_ticker} อยู่ใน Watchlist แล้ว")

                # กราฟแท่งเทียน
                if raw_df is not None:
                    st.markdown(f"#### {UI_LANG_MAP['chart_title_single']}")
                    st.markdown(f'<div class="chart-header-badge">{single_ticker} | ล่าสุด: ${res.get("Price ($)", 0)} (RSI: {res.get("RSI", 0)})</div>', unsafe_allow_html=True)
                    fig = create_ta_chart(raw_df, single_ticker, res)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"chart_single_{single_ticker}")

                st.markdown("---")
                
                # กล่อง Compact Board
                st.markdown(f"#### {UI_LANG_MAP['analysis_title']}")
                badge_class = res.get('badge_class', 'badge-trend-side')
                badge_label = res.get('badge_label', '〰️ สะสมแรง')
                pat_name = res.get('pattern_name', 'สร้างฐานสะสมกำลัง.png')
                pat_score = res.get('pattern_score', 75.0)

                st.markdown(f"""
                <div class="compact-board">
                    <div class="price-banner">
                        <div class="price-val-box">
                            <span style="font-size:0.8rem; color:#94A3B8; font-weight:600;">💰 ราคา:</span>
                            <span class="price-main">${res.get('Price ($)', 0)}</span>
                        </div>
                        <div class="price-badge-group">
                            <span class="price-badge {badge_class}">{badge_label}</span>
                            <span class="price-badge badge-ai-box">🤖 AI Pattern: {pat_name} ({pat_score}%)</span>
                            <span class="price-badge badge-rsi">RSI: {res.get('RSI', 0)}</span>
                            <span class="price-badge badge-dist">ห่างรับ 1: {res.get('Dist_Sup (%)', '0%')}</span>
                        </div>
                    </div>
                    <div class="snr-grid">
                        <div class="snr-card" style="border-left: 3px solid #22C55E;">
                            <div class="snr-card-title c-green">🛡️ แนวรับ (Support)</div>
                            <div class="snr-row"><span class="snr-lbl">รับ 1 (สวิงใกล้สุด)</span><span class="snr-num c-green">${res.get('Support 1 ($)', 0)}</span></div>
                            <div class="snr-row"><span class="snr-lbl">รับ 2 (ฐานหลัก)</span><span class="snr-num c-lightgreen">${res.get('Support 2 ($)', 0)}</span></div>
                            <div class="snr-row"><span class="snr-lbl">รับ 3 (โครงสร้างใหญ่)</span><span class="snr-num c-lightgreen">${res.get('Support 3 ($)', 0)}</span></div>
                        </div>
                        <div class="snr-card" style="border-left: 3px solid #EF4444;">
                            <div class="snr-card-title c-red">⚡ แนวต้าน (Resistance)</div>
                            <div class="snr-row"><span class="snr-lbl">ต้าน 1</span><span class="snr-num c-red">${res.get('Resist 1 ($)', 0)}</span></div>
                            <div class="snr-row"><span class="snr-lbl">ต้าน 2</span><span class="snr-num c-orange">${res.get('Resist 2 ($)', 0)}</span></div>
                            <div class="snr-row"><span class="snr-lbl">ต้าน 3</span><span class="snr-num c-yellow">${res.get('Resist 3 ($)', 0)}</span></div>
                            <div class="snr-row"><span class="snr-lbl">ต้าน 4 (สูงสุด)</span><span class="snr-num c-darkred">${res.get('Resist 4 ($)', 0)}</span></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # การ์ดกลยุทธ์แนวรับแนวตั้ง
                st.markdown("#### 🎯 กลยุทธ์แบ่งไม้เข้าซื้อ & ประเมินความแข็งแรงของแนวรับ")
                curr_p = res.get('Price ($)', 1.0)
                dist_s1 = ((res.get('Support 1 ($)', 0) - curr_p) / curr_p) * 100
                dist_s2 = ((res.get('Support 2 ($)', 0) - curr_p) / curr_p) * 100
                dist_s3 = ((res.get('Support 3 ($)', 0) - curr_p) / curr_p) * 100

                st.markdown(f"""
                <div class="strategy-card" style="border-left: 4px solid #22C55E;">
                    <div class="strat-header">
                        <div>
                            <span class="strat-title">🛡️ แนวรับ 1 (สวิงโลว์ใกล้สุด)</span>
                            <span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">({dist_s1:+.2f}%)</span>
                        </div>
                        <span class="strat-price" style="color:#22C55E;">${res.get('Support 1 ($)', 0)}</span>
                    </div>
                    <div class="strat-body">
                        <div><span class="strat-sub">ความแข็งแรง:</span> <span class="strat-val">⭐⭐ ปานกลาง</span></div>
                        <div><span class="strat-sub">กลยุทธ์:</span> <span class="strat-val c-green">25% (ไม้หยั่งเชิง / ดูแรงเด้ง)</span></div>
                    </div>
                </div>

                <div class="strategy-card" style="border-left: 4px solid #16A34A;">
                    <div class="strat-header">
                        <div>
                            <span class="strat-title">🛡️ แนวรับ 2 (ฐานสะสมหลัก)</span>
                            <span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">({dist_s2:+.2f}%)</span>
                        </div>
                        <span class="strat-price" style="color:#4ADE80;">${res.get('Support 2 ($)', 0)}</span>
                    </div>
                    <div class="strat-body">
                        <div><span class="strat-sub">ความแข็งแรง:</span> <span class="strat-val">⭐⭐⭐⭐ แข็งแกร่ง</span></div>
                        <div><span class="strat-sub">กลยุทธ์:</span> <span class="strat-val c-lightgreen">35% (ไม้หลักสะสมของ)</span></div>
                    </div>
                </div>

                <div class="strategy-card" style="border-left: 4px solid #15803D;">
                    <div class="strat-header">
                        <div>
                            <span class="strat-title">🛡️ แนวรับ 3 (ฐานโครงสร้างใหญ่)</span>
                            <span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">({dist_s3:+.2f}%)</span>
                        </div>
                        <span class="strat-price" style="color:#86EFAC;">${res.get('Support 3 ($)', 0)}</span>
                    </div>
                    <div class="strat-body">
                        <div><span class="strat-sub">ความแข็งแรง:</span> <span class="strat-val">⭐⭐⭐⭐⭐ แข็งแกร่งมาก</span></div>
                        <div><span class="strat-sub">กลยุทธ์:</span> <span class="strat-val" style="color:#86EFAC;">40% (ไม้สะสมลึก / กลับตัวใหญ่)</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ข่าวสารล่าสุด
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

                # งบการเงินย้อนหลัง
                st.markdown("#### 💰 กำไรสุทธิ 3 ไตรมาสล่าสุด")
                if df_profit is not None:
                    c_table, c_chart = st.columns(2)
                    with c_table:
                        st.dataframe(df_profit, use_container_width=True, hide_index=True, height=125)
                    with c_chart:
                        bar_colors = ['#22C55E' if v >= 0 else '#EF4444' for v in df_profit['Net Income (M$)']]
                        fig_profit = go.Figure(data=[go.Bar(
                            x=df_profit['Quarter End'],
                            y=df_profit['Net Income (M$)'],
                            marker_color=bar_colors
                        )])
                        fig_profit.update_layout(
                            margin=dict(l=8, r=8, t=8, b=8),
                            height=125,
                            template='plotly_dark',
                            xaxis_title="",
                            yaxis_title="M$"
                        )
                        st.plotly_chart(fig_profit, use_container_width=True, config={'displayModeBar': False}, key=f"chart_profit_{single_ticker}")
                else:
                    st.warning("ไม่พบข้อมูลกำไรสุทธิย้อนหลัง")

                st.markdown("---")
                
                # โครงสร้างผู้ถือหุ้น
                summary_text = res.get('summaryTh', 'N/A')
                shares_tot = res.get('sharesOutstanding', 'N/A')
                inst_pct = res.get('institutionalHeld', 'N/A')
                insider_pct = res.get('insiderHeld', 'N/A')
                retail_pct = res.get('retailHeld', 'N/A')
                expander_title = UI_LANG_MAP.get('expander_business_summary', "📖 สรุปธุรกิจ & โครงสร้างผู้ถือหุ้น (แปลไทยอัตโนมัติ)")

                with st.expander(expander_title, expanded=True):
                    st.markdown(f"""
                    <div class="fin-card">
                        <b style="color: #60A5FA; font-size: 0.88rem;">📊 โครงสร้างผู้ถือหุ้น & ข้อมูลบริษัท:</b>
                        <div style="color: #F8FAFC; line-height: 1.7; margin-top: 4px; font-size: 0.82rem;">
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
                         st.markdown(f'<div class="biz-summary"><b>[ลักษณะการทำธุรกิจ]</b><br>{summary_text}</div>', unsafe_allow_html=True)
                    else:
                         st.warning("⚠️ ไม่พบข้อมูลสรุปธุรกิจสำหรับหุ้นตัวนี้")
            else:
                st.error(f"❌ ไม่พบข้อมูลสัญลักษณ์หุ้น **{single_ticker}** ในระบบ กรุณาตรวจสอบชื่อ Ticker อีกครั้ง")


# --- TAB 2: สแกนคัดหุ้นทั้งตลาด ---
with tab2:
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน (NASDAQ, NYSE, AMEX)")
    
    # ระบบ Auto-Unlock ป้องกันสถานะค้างเกิน 2 นาที
    if server_state["is_scanning"] and server_state.get("scan_start_time"):
        elapsed_scan = (datetime.now() - server_state["scan_start_time"]).total_seconds()
        if elapsed_scan > 120:
            server_state["is_scanning"] = False
            server_state["scan_start_time"] = None

    # เมนูเลือกขอบเขตการสแกน
    scan_scope = st.radio(
        "🎯 เลือกขอบเขตและจำนวนหุ้นที่จะสแกน:",
        ["⚡ หุ้นผู้นำตลาด S&P 500 & Top Tech (500 ตัวจริง - สแกนเร็ว 15 วิ)",
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

    if is_busy:
        st.warning("⏳ **ระบบกำลังประมวลผลการสแกนอยู่** กรุณารอสักครู่ (หากค้างสามารถกดปุ่ม 'ปลดล็อก & รีเซ็ตระบบ' ได้ทันที)")

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
        server_state["is_scanning"] = True
        server_state["scan_start_time"] = datetime.now()
        status_text = st.empty()
        status_text.info(UI_LANG_MAP['status_preparing_tickers'])
        
        stock_list = get_us_stock_tickers(scope_code)
        total_stocks = len(stock_list)
        progress_bar = st.progress(0)
        
        results = []
        count = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
                futures = {executor.submit(check_ma_snr_combo, ticker, False): ticker for ticker in stock_list}
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
            df_result_display = pd.DataFrame([item['res_data'] for item in results])[[
                'Ticker', 'Price ($)', 'status_text', 'pattern_name', 'Support 1 ($)', 'Support 2 ($)', 'Support 3 ($)', 'RSI', 
                'Resist 1 ($)', 'Resist 2 ($)', 'Resist 3 ($)', 'Resist 4 ($)', 
                'Volume', 'Date'
            ]]
            server_state["latest_df"] = df_result_display
        st.rerun()

    if server_state["latest_results"]:
        results = server_state["latest_results"]
        df_result_display = server_state["latest_df"]

        if server_state.get("last_scanned_at"):
            elapsed_thai = get_time_elapsed_thai(server_state.get("last_scanned_dt"))
            st.info(f"🕒 ผลการสแกนล่าสุดของเซิร์ฟเวอร์ ณ เวลา: **{server_state['last_scanned_at']}**{elapsed_thai} (ทุกคนในระบบสามารถดูร่วมกันได้ทันที)")

        st.markdown("---")
        st.subheader('📸 แกลเลอรี่กราฟหุ้นทรงสวย (พร้อมรายละเอียดบริษัทและ AI Pattern Match)')
        
        items_per_page = 6
        total_pages = max(1, (len(results) + items_per_page - 1) // items_per_page)
        page_num = st.selectbox("เลือกหน้าแสดงผลกราฟ:", range(1, int(total_pages) + 1), key="pagination_select")
        
        start_idx = (page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_page_items = results[start_idx:end_idx]

        for row_idx in range(0, len(current_page_items), 2):
            cols = st.columns(2)
            for c_offset in range(2):
                item_idx = row_idx + c_offset
                if item_idx < len(current_page_items):
                    item = current_page_items[item_idx]
                    res_data = item['res_data']
                    ticker_found = res_data.get('Ticker', '')
                    raw_df_found = item.get('raw_df')
                    status_lbl = res_data.get('status_text', '')
                    pat_found = res_data.get('pattern_name', 'สร้างฐานสะสมกำลัง.png')
                    pat_sc_found = res_data.get('pattern_score', 75.0)

                    with cols[c_offset]:
                        with st.container():
                            st.markdown(f'<p style="font-size:0.92rem; font-weight:bold; color:#60A5FA; margin-bottom:0px;">🟢 {ticker_found} | {status_lbl}</p>', unsafe_allow_html=True)
                            st.caption(f"Support 1: ${res_data.get('Support 1 ($)', 0)} | ต้าน 1: ${res_data.get('Resist 1 ($)', 0)} | RSI: {res_data.get('RSI', 0)}")
                            
                            if ticker_found not in st.session_state.watchlist:
                                if st.button(f"⭐ บันทึก {ticker_found} เข้า Watchlist", key=f"btn_gal_wl_{ticker_found}_{page_num}_{item_idx}"):
                                    st.session_state.watchlist.append(ticker_found)
                                    st.rerun()
                            
                            if raw_df_found is not None:
                                st.markdown(f'<div class="chart-header-badge">{ticker_found} | ล่าสุด: ${res_data.get("Price ($)", 0)} (RSI: {res_data.get("RSI", 0)})</div>', unsafe_allow_html=True)
                                fig_gallery = create_ta_chart(raw_df_found, ticker_found, res_data)
                                if fig_gallery:
                                    st.plotly_chart(fig_gallery, use_container_width=True, config=PLOTLY_CONFIG, key=f"gallery_chart_{ticker_found}_{page_num}_{item_idx}")
                                
                                st.markdown(f'<div class="pattern-box" style="font-size:0.72rem; padding:3px 6px;">🤖 AI Pattern: {pat_found} ({pat_sc_found}%)</div>', unsafe_allow_html=True)
                            else:
                                st.warning("ไม่พบข้อมูลกราฟ")
                                
                            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 ตารางสรุปสัญญาณราคาหุ้นทรงสวยประจำวัน")
        st.dataframe(df_result_display, use_container_width=True, hide_index=True, height=200)
        st.download_button(
            label='📥 ดาวน์โหลด Watchlist วันนี้ (CSV)',
            data=df_result_display.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f'us_watchlist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            key="btn_download_csv"
        )


# --- TAB 3: Watchlist ส่วนตัว ---
with tab3:
    st.markdown("### ⭐ รายชื่อหุ้นใน Watchlist ส่วนตัวของคุณ")
    if st.session_state.watchlist:
        if st.button("🗑️ ล้างรายชื่อ Watchlist ทั้งหมด", key="btn_clear_wl_all"):
            st.session_state.watchlist = []
            st.rerun()
        
        st.write(f"หุ้นที่คุณติดตามอยู่ทั้งหมด: {', '.join(st.session_state.watchlist)}")
        st.markdown("---")
        
        for w_ticker in st.session_state.watchlist:
            try:
                df_w = fetch_stock_history_dual(w_ticker)
                if df_w is not None and not df_w.empty:
                    curr_price = round(float(df_w['close'].iloc[-1]), 2)
                    prev_close = float(df_w['close'].iloc[-2]) if len(df_w) > 1 else curr_price
                    change = round(((curr_price - prev_close) / prev_close) * 100, 2)
                    
                    st.markdown(f"""
                    <div class="fin-card">
                        <b>📌 {w_ticker}</b> | ราคาล่าสุด: <b>${curr_price}</b> ({change:+.2f}%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"📌 **{w_ticker}** | ไม่สามารถดึงข้อมูลราคาได้ในขณะนี้")
            except Exception:
                st.warning(f"📌 **{w_ticker}** | เกิดข้อผิดพลาดในการเชื่อมต่อข้อมูล")
    else:
        st.info("ยังไม่มีหุ้นใน Watchlist ส่วนตัว ลองค้นหาหุ้นรายตัวแล้วกดปุ่มเพิ่มได้เลยครับ!")
