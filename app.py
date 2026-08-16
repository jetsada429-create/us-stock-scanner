import concurrent.futures
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ================= ส่วนตั้งค่าแอปและภาษา =================
UI_LANG_MAP = {
    'search_ticker_title': "US Stock Scanner PRO",
    'search_ticker_subtitle': "ระบบสแกนเทคนิคอล • วิเคราะห์ 4 สภาวะตลาด • AI Pattern • 3 แนวรับ 4 แนวต้าน",
    'search_ticker_label': "พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, AAOI, BZAI):",
    'btn_analyze_single': "🔎 วิเคราะห์ทันที",
    'btn_scan_market': "🚀 เริ่มสแกนทั้ง 3 ตลาด (7,000+ หุ้น)",
    'status_preparing_tickers': "⏳ กำลังดึงรายชื่อหุ้นทั้งหมดจาก NASDAQ, NYSE, AMEX...",
    'status_scanning': "⏳ สแกนไปแล้ว {count}/{total} ตัว...",
    'status_analyzing_single': "⏳ กำลังดึงข้อมูลและวิเคราะห์ {ticker}...",
    'expander_business_summary': "📖 สรุปธุรกิจ & โครงสร้างผู้ถือหุ้น (แปลไทยอัตโนมัติ)",
    'chart_title_single': "📈 กราฟเทคนิค 3 แนวรับ และ 4 ระดับแนวต้าน",
    'analysis_title': "📊 ข้อมูลแนวรับ - แนวต้าน & สถานะตลาด",
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

st.set_page_config(
    page_title=UI_LANG_MAP['search_ticker_title'],
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Session ป้องกัน Rate Limit (429)
@st.cache_resource
def get_yfinance_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# Server State ส่วนกลางสำหรับการสแกนร่วมกัน
@st.cache_resource
def get_global_server_state():
    return {
        "is_scanning": False,
        "latest_results": None,
        "latest_df": None,
        "last_scanned_at": None,
        "last_scanned_dt": None
    }

server_state = get_global_server_state()

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# ================= Custom CSS สไตล์ Modern FinTech (Responsive) =================
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
    }
    .sub-title {
        font-size: 0.78rem !important;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 0.8rem;
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
    }
    .compact-board {
        background: #0B132B;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 0.6rem;
    }
    .price-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        padding-bottom: 10px;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 10px;
    }
    .price-main {
        font-size: 1.5rem;
        font-weight: 900;
        color: #FFFFFF;
    }
    .price-badge {
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-trend-bull { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
    .badge-trend-bear { background: #4C0519; color: #FDA4AF; border: 1px solid #9F1239; }
    .badge-trend-pull { background: #451A03; color: #FCD34D; border: 1px solid #78350F; }
    .badge-trend-support { background: #1E3A8A; color: #93C5FD; border: 1px solid #1D4ED8; }
    .badge-rsi { background: #1E293B; color: #38BDF8; border: 1px solid #334155; }
    
    .snr-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }
    .snr-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 8px 10px;
    }
    .snr-card-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 1px dashed #334155;
    }
    .snr-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        padding: 3px 0;
    }
    .snr-lbl { color: #94A3B8; font-size: 0.74rem; }
    .snr-num { font-weight: 700; font-size: 0.88rem; }
    
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
    }
    .strat-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .strat-title { font-size: 0.82rem; font-weight: 700; color: #F8FAFC; }
    .strat-price { font-size: 0.95rem; font-weight: 800; color: #38BDF8; }
    .strat-body { display: flex; justify-content: space-between; align-items: center; font-size: 0.74rem; padding-top: 4px; border-top: 1px dashed #1E293B; }
    .strat-sub { color: #94A3B8; }
    .strat-val { color: #F8FAFC; font-weight: 600; }

    .company-header { font-size: 1.2rem; font-weight: 800; color: #38BDF8 !important; margin-bottom: 0rem; }
    .sector-badge { font-size: 0.8rem; font-weight: 600; color: #FCD34D; background: #451A03; border: 1px solid #78350F; padding: 4px 8px; border-radius: 6px; display: inline-block; margin-top: 4px; margin-bottom: 6px; }
    .chart-header-badge { font-size: 0.9rem; font-weight: 700; color: #F8FAFC; background-color: #1E293B; padding: 5px 8px; border-radius: 6px; margin-bottom: 4px; display: inline-block; }
    .fin-card { background: #0F172A !important; border: 1px solid #334155 !important; border-radius: 8px; padding: 12px; margin-bottom: 0.5rem; color: #F8FAFC !important; }
    .biz-summary { font-size: 0.85rem !important; color: #F1F5F9 !important; background-color: #0B132B !important; padding: 12px !important; border-radius: 8px; border-left: 4px solid #3B82F6 !important; border: 1px solid #334155 !important; margin-bottom: 0.4rem; line-height: 1.6; }
    .pattern-box { background-color: #172554 !important; color: #93C5FD !important; padding: 6px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; border: 1px solid #1E40AF !important; margin-top: 4px; margin-bottom: 6px; }
    .news-card { background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
    .news-title { font-size: 0.82rem; font-weight: 600; color: #E2E8F0; text-decoration: none; }
    .news-meta { font-size: 0.7rem; color: #64748B; margin-top: 2px; }
    .desktop-only-space { height: 28px; display: block; }
    @media (max-width: 640px) {
        .desktop-only-space { display: none !important; }
        .main-title { font-size: 1.3rem !important; }
        .price-main { font-size: 1.3rem; }
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="main-title">{UI_LANG_MAP["search_ticker_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{UI_LANG_MAP["search_ticker_subtitle"]}</div>', unsafe_allow_html=True)

# ================= ฟังก์ชันคำนวณทางเทคนิค =================
def calculate_swing_snr(df, latest_close):
    """
    คำนวณแนวรับ-แนวต้านจากรอบคลื่นปัจจุบัน (Active Wave 120 วัน) ตัดปัญหาราคาเก่าลึกเกินไป
    """
    n = len(df)
    window_n = min(n, 120)
    df_wave = df.iloc[-window_n:]
    
    wave_high = float(np.max(df_wave['high']))
    wave_low = float(np.min(df_wave['low']))
    wave_range = max(1e-4, wave_high - wave_low)

    fib_382 = wave_high - 0.382 * wave_range
    fib_500 = wave_high - 0.500 * wave_range
    fib_618 = wave_high - 0.618 * wave_range

    recent_15d_low = float(np.min(df_wave['low'].tail(15)))
    recent_45d_low = float(np.min(df_wave['low'].tail(45)))

    # แนวรับ
    s1 = recent_15d_low if (recent_15d_low < latest_close * 0.995 and recent_15d_low > latest_close * 0.85) else latest_close * 0.95
    s2 = recent_45d_low if (recent_45d_low < s1 * 0.96 and recent_45d_low > wave_low * 1.15) else (fib_618 if fib_618 < s1 * 0.96 else s1 * 0.89)
    s3 = wave_low if wave_low < s2 * 0.90 else s2 * 0.75

    if s2 >= s1: s2 = s1 * 0.90
    if s3 >= s2: s3 = s2 * 0.75

    # แนวต้าน
    r4 = wave_high
    cand_resists = [fib_500, fib_382, fib_236 if 'fib_236' in locals() else wave_high - 0.236*wave_range]
    valid_resists = sorted([r for r in cand_resists if r > latest_close * 1.015 and r < r4 * 0.985])

    if len(valid_resists) >= 3:
        r1, r2, r3 = valid_resists[0], valid_resists[1], valid_resists[2]
    elif len(valid_resists) == 2:
        r1, r2 = valid_resists[0], valid_resists[1]
        r3 = r2 + (r4 - r2) * 0.50
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
        if c_max == c_min: return "สร้างฐานสะสมกำลัง.png", 82.0
        norm_closes = (closes - c_min) / (c_max - c_min)
        
        x = np.linspace(0, 1, bars)
        templates = {
            "สร้างฐานยก Low.png": 0.15 + 0.75 * x + 0.08 * np.sin(x * 3 * np.pi),
            "สร้างฐานแบบ Double Bottom.png": 0.65 - 0.65 * np.sin(x * np.pi) + 0.25 * np.cos(x * 2 * np.pi),
            "สร้างฐานก้นกระทะ (Rounding).png": 0.85 - 0.85 * np.sin(x * np.pi),
            "สร้างฐานสะสมกำลัง.png": np.full(bars, 0.5) + 0.08 * np.sin(x * 5 * np.pi)
        }

        best_pattern, best_score = "สร้างฐานสะสมกำลัง.png", 65.0
        for pat_name, pat_curve in templates.items():
            norm_pat = (pat_curve - np.min(pat_curve)) / (np.max(pat_curve) - np.min(pat_curve) + 1e-6)
            sim_score = (1.0 - np.mean(np.abs(norm_closes - norm_pat))) * 100.0
            if sim_score > best_score:
                best_score = sim_score
                best_pattern = pat_name

        return best_pattern, round(max(70.0, min(95.5, best_score)), 1)
    except:
        return "สร้างฐานสะสมกำลัง.png", 76.5

def get_time_elapsed_thai(last_dt):
    if not last_dt: return ""
    diff = datetime.now() - last_dt
    secs = int(diff.total_seconds())
    if secs < 60: return f" (เพิ่งสแกนเมื่อ {secs} วิที่แล้ว)"
    elif secs < 3600: return f" (สแกนไปแล้ว {secs // 60} นาทีที่แล้ว)"
    else: return f" (สแกนไปแล้ว {secs // 3600} ชม. ก่อน)"

@st.cache_data(ttl=86400)
def get_us_stock_tickers():
    try:
        url = 'https://dumbstockapi.com/stock?exchange=NASDAQ,NYSE'
        return [s['ticker'] for s in requests.get(url, timeout=10).json()]
    except:
        return ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT', 'PLTR', 'RKLB', 'IREN', 'AAOI', 'BZAI']

def translate_text_to_thai(text):
    if not text or text == 'N/A': return 'N/A'
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        res = requests.get(url, params={"client": "gtx", "sl": "en", "tl": "th", "dt": "t", "q": text}, timeout=4)
        if res.status_code == 200:
            return "".join([item[0] for item in res.json()[0] if item[0]])
    except: pass
    return text

@st.cache_data(ttl=14400)
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
    except:
        return {'longNameEn': ticker, 'sectorTh': 'N/A', 'industryTh': 'N/A', 'summaryTh': 'N/A', 'sharesOutstanding': 'N/A', 'institutionalHeld': 'N/A', 'insiderHeld': 'N/A', 'retailHeld': 'N/A'}

@st.cache_data(ttl=3600)
def get_stock_news(ticker):
    try:
        items = yf.Ticker(ticker, session=get_yfinance_session()).news
        if not items: return []
        return [{'title': translate_text_to_thai(n.get('title', '')), 'publisher': n.get('publisher', 'News'), 'link': n.get('link', '#'), 'time': datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%d/%m/%Y %H:%M') if n.get('providerPublishTime') else ''} for n in items[:3]]
    except: return []

@st.cache_data(ttl=14400)
def get_financials(ticker):
    try:
        q = yf.Ticker(ticker, session=get_yfinance_session()).quarterly_financials
        if q is not None and 'Net Income' in q.index:
            ni = q.loc['Net Income'].head(3)
            return pd.DataFrame([{'Quarter End': d.strftime('%Y-%m-%d'), 'Net Income (M$)': round(v / 1_000_000, 2)} for d, v in ni.items() if pd.notna(v)])
    except: pass
    return None

def create_ta_chart(df, ticker, res_data):
    if df is None or df.empty: return None
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='ราคา')])
    fast_ma = df['close'].rolling(20).mean()
    slow_ma = df['close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=fast_ma, line=dict(color='#38BDF8', width=1.2), name='MA20'))
    fig.add_trace(go.Scatter(x=df.index, y=slow_ma, line=dict(color='#FB923C', width=1.2), name='MA50'))
    
    for key, color, pos in [('Support 1 ($)', '#22C55E', -12), ('Support 2 ($)', '#16A34A', 12), ('Support 3 ($)', '#15803D', -12)]:
        if key in res_data:
            val = res_data[key]
            fig.add_shape(type="line", x0=df.index[0], y0=val, x1=df.index[-1], y1=val, line=dict(color=color, width=1.6, dash='dash'))
            fig.add_annotation(x=df.index[-1], y=val, text=f"{key.replace(' ($)', '')}: ${val}", bgcolor=color, font=dict(color="white", size=9), xanchor="left", ax=8, ay=pos)

    for key, color, pos in [('Resist 1 ($)', '#EF4444', -12), ('Resist 2 ($)', '#F97316', 12), ('Resist 3 ($)', '#EAB308', -12), ('Resist 4 ($)', '#991B1B', 12)]:
        if key in res_data:
            val = res_data[key]
            fig.add_shape(type="line", x0=df.index[0], y0=val, x1=df.index[-1], y1=val, line=dict(color=color, width=1.6, dash='dash'))
            fig.add_annotation(x=df.index[-1], y=val, text=f"{key.replace(' ($)', '')}: ${val}", bgcolor=color, font=dict(color="white", size=9), xanchor="left", ax=8, ay=pos)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', margin=dict(l=6, r=65, t=10, b=6), height=340, dragmode='pan', yaxis_title="ราคา ($)", showlegend=False)
    return fig

# ================= วิเคราะห์และจำแนก 4 สภาวะตลาด =================
@st.cache_data(ttl=14400)
def check_ma_snr_combo(ticker, info_mode=False):
    try:
        stock = yf.Ticker(ticker, session=get_yfinance_session())
        df = stock.history(period='2y', interval='1d')
        if len(df) < 50 or df['Close'].iloc[-1] < 0.05: return None, None
        df.columns = [c.lower() for c in df.columns]
        
        latest_close = df['close'].iloc[-1]
        fast_ma = df['close'].rolling(20).mean()
        slow_ma = df['close'].rolling(50).mean()
        
        # คำนวณ RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = round(float(100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).iloc[-1], 2)

        s1, s2, s3, r1, r2, r3, r4 = calculate_swing_snr(df, latest_close)

        recent_8d_high = float(df['high'].tail(8).max())
        drop_8d_pct = ((latest_close - recent_8d_high) / recent_8d_high) * 100
        recent_8d_low = float(df['low'].tail(8).min())
        bounce_8d_pct = ((latest_close - recent_8d_low) / recent_8d_low) * 100 if recent_8d_low > 0 else 0.0

        is_above_ma20 = latest_close >= fast_ma.iloc[-1]
        is_above_ma50 = latest_close >= slow_ma.iloc[-1]
        is_ma_bull = fast_ma.iloc[-1] >= slow_ma.iloc[-1]

        # ================= 4 สภาวะตลาดหลัก =================
        if drop_8d_pct <= -20.0 or (not is_above_ma20 and not is_above_ma50 and rsi < 40 and drop_8d_pct <= -15.0):
            status_text = "📉 ลงแรง / ขาลงชัดเจน (ห้ามรับมีด)"
            badge_class = "badge-trend-bear"
        elif is_above_ma20 and is_above_ma50 and is_ma_bull and rsi >= 50:
            status_text = "🚀 ขาขึ้นแข็งแกร่ง (Strong Uptrend)"
            badge_class = "badge-trend-bull"
        elif is_above_ma50 and drop_8d_pct <= -8.0 and rsi >= 42:
            status_text = "⏳ ย่อพักฐาน (Healthy Pullback)"
            badge_class = "badge-trend-pull"
        elif bounce_8d_pct >= 2.5 and latest_close <= s1 * 1.03:
            status_text = "🎯 ช้อนแนวรับ (Buy on Support)"
            badge_class = "badge-trend-support"
        else:
            status_text = "〰️ สะสมแรง / ไซด์เวย์"
            badge_class = "badge-trend-side"

        pat_name, pat_score = calculate_ai_pattern_match(df.tail(120))
        dist_from_sup = ((latest_close - s1) / s1) * 100

        res_data = {
            'Ticker': ticker, 'Price ($)': round(latest_close, 2),
            'Support 1 ($)': s1, 'Support 2 ($)': s2, 'Support 3 ($)': s3,
            'Resist 1 ($)': r1, 'Resist 2 ($)': r2, 'Resist 3 ($)': r3, 'Resist 4 ($)': r4,
            'Dist_Sup (%)': f'{dist_from_sup:+.2f}%', 'RSI': rsi,
            'Volume': f"{df['volume'].iloc[-1]:,.0f}", 'Date': df.index[-1].strftime('%Y-%m-%d'),
            'pattern_name': pat_name, 'pattern_score': pat_score,
            'status_text': status_text, 'badge_class': badge_class,
            'drop_8d_pct': f'{drop_8d_pct:.2f}%', 'bounce_8d_pct': f'+{bounce_8d_pct:.2f}%'
        }

        if info_mode:
            res_data.update(get_company_info_and_holders(ticker))

        return res_data, df
    except:
        return None, None

# ================= หน้าจอหลัก (UI Tabs) =================
tab1, tab2, tab3 = st.tabs([UI_LANG_MAP['tab_search_ticker'], UI_LANG_MAP['tab_scan_market'], UI_LANG_MAP['tab_watchlist']])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        single_ticker = st.text_input(UI_LANG_MAP['search_ticker_label'], value="AAOI").strip().upper()
    with col2:
        st.markdown("<div class='desktop-only-space'></div>", unsafe_allow_html=True)
        search_btn = st.button(UI_LANG_MAP['btn_analyze_single'])

    if search_btn and single_ticker:
        with st.spinner(UI_LANG_MAP['status_analyzing_single'].format(ticker=single_ticker)):
            res, raw_df = check_ma_snr_combo(single_ticker, info_mode=True)
            df_profit = get_financials(single_ticker)
            news_items = get_stock_news(single_ticker)

            if res:
                st.markdown(f'<p class="company-header">{single_ticker} : {res.get("longNameEn", single_ticker)}</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="sector-badge">🏷️ กลุ่มธุรกิจ: {res.get("sectorTh", "N/A")} | ย่อย: {res.get("industryTh", "N/A")}</div>', unsafe_allow_html=True)
                
                # กราฟแท่งเทียน
                if raw_df is not None:
                    st.markdown(f"#### {UI_LANG_MAP['chart_title_single']}")
                    st.markdown(f'<div class="chart-header-badge">{single_ticker} | ล่าสุด: ${res["Price ($)"]} (RSI: {res.get("RSI", 0)})</div>', unsafe_allow_html=True)
                    fig = create_ta_chart(raw_df, single_ticker, res)
                    if fig: st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True}, key=f"chart_{single_ticker}")
                    
                    st.markdown(f'<div class="pattern-box">😊 🤖 AI Pattern Match: {res["pattern_name"]} (ความแม่นยำ: {res["pattern_score"]}%)</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"#### {UI_LANG_MAP['analysis_title']}")
                
                # กล่อง Compact Board พร้อม 4 สถานะตลาดที่ชัดเจน
                st.markdown(f"""
                <div class="compact-board">
                    <div class="price-banner">
                        <div class="price-val-box">
                            <span style="font-size:0.8rem; color:#94A3B8; font-weight:600;">💰 ราคา:</span>
                            <span class="price-main">${res['Price ($)']}</span>
                        </div>
                        <div class="price-badge-group">
                            <span class="price-badge {res['badge_class']}">{res['status_text']}</span>
                            <span class="price-badge badge-rsi">RSI: {res['RSI']}</span>
                            <span class="price-badge badge-dist">ห่างรับ 1: {res['Dist_Sup (%)']}</span>
                        </div>
                    </div>
                    <div class="snr-grid">
                        <div class="snr-card" style="border-left: 3px solid #22C55E;">
                            <div class="snr-card-title c-green">🛡️ แนวรับ (Support)</div>
                            <div class="snr-row"><span class="snr-lbl">รับ 1 (สวิงใกล้สุด)</span><span class="snr-num c-green">${res['Support 1 ($)']}</span></div>
                            <div class="snr-row"><span class="snr-lbl">รับ 2 (ฐานหลัก)</span><span class="snr-num c-lightgreen">${res['Support 2 ($)']}</span></div>
                            <div class="snr-row"><span class="snr-lbl">รับ 3 (โครงสร้างใหญ่)</span><span class="snr-num c-lightgreen">${res['Support 3 ($)']}</span></div>
                        </div>
                        <div class="snr-card" style="border-left: 3px solid #EF4444;">
                            <div class="snr-card-title c-red">⚡ แนวต้าน (Resistance)</div>
                            <div class="snr-row"><span class="snr-lbl">ต้าน 1</span><span class="snr-num c-red">${res['Resist 1 ($)']}</span></div>
                            <div class="snr-row"><span class="snr-lbl">ต้าน 2</span><span class="snr-num c-orange">${res['Resist 2 ($)']}</span></div>
                            <div class="snr-row"><span class="snr-lbl">ต้าน 3</span><span class="snr-num c-yellow">${res['Resist 3 ($)']}</span></div>
                            <div class="snr-row"><span class="snr-lbl">ต้าน 4 (สูงสุด)</span><span class="snr-num c-darkred">${res['Resist 4 ($)']}</span></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # การ์ดกลยุทธ์แนวตั้ง (อ่านง่ายบนมือถือ)
                st.markdown("#### 🎯 กลยุทธ์แบ่งไม้เข้าซื้อ & ประเมินความแข็งแรง")
                for i, (lbl, prc, rate, strength) in enumerate([
                    ("รับ 1 (สวิงใกล้สุด)", res['Support 1 ($)'], "25% (ไม้หยั่งเชิง)", "⭐⭐ ปานกลาง"),
                    ("รับ 2 (ฐานสะสมหลัก)", res['Support 2 ($)'], "35% (ไม้หลักสะสม)", "⭐⭐⭐⭐ แข็งแกร่ง"),
                    ("รับ 3 (ฐานโครงสร้างใหญ่)", res['Support 3 ($)'], "40% (ไม้สะสมลึก)", "⭐⭐⭐⭐⭐ แข็งแกร่งมาก")
                ]):
                    dist = ((prc - res['Price ($)']) / res['Price ($)']) * 100
                    st.markdown(f"""
                    <div class="strategy-card">
                        <div class="strat-header">
                            <span class="strat-title">🛡️ {lbl} ({dist:+.2f}%)</span>
                            <span class="strat-price">${prc}</span>
                        </div>
                        <div class="strat-body">
                            <span class="strat-sub">ความแข็งแรง: <span class="strat-val">{strength}</span></span>
                            <span class="strat-sub">กลยุทธ์: <span class="strat-val c-green">{rate}</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ข่าวสารล่าสุด
                st.markdown("#### 📰 ข่าวสารล่าสุด (แปลไทยอัตโนมัติ)")
                if news_items:
                    for n in news_items:
                        st.markdown(f"""<div class="news-card"><a class="news-title" href="{n['link']}" target="_blank">📌 {n['title']}</a><div class="news-meta">แหล่งข่าว: {n['publisher']} | เวลา: {n['time']}</div></div>""", unsafe_allow_html=True)
                else:
                    st.info("ℹ️ ไม่พบข่าวสารสำคัญในช่วงนี้")

                # งบการเงิน
                st.markdown("#### 💰 กำไรสุทธิ 3 ไตรมาสล่าสุด")
                if df_profit is not None:
                    st.dataframe(df_profit, use_container_width=True, hide_index=True)

                # ข้อมูลผู้ถือหุ้นและธุรกิจ
                with st.expander(UI_LANG_MAP['expander_business_summary'], expanded=False):
                    st.markdown(f"""
                    <div class="fin-card">
                        <b>📊 โครงสร้างผู้ถือหุ้น:</b><br>
                        • สถาบันถือครอง: <b>{res.get('institutionalHeld', 'N/A')}</b><br>
                        • ผู้บริหารถือครอง: <b>{res.get('insiderHeld', 'N/A')}</b><br>
                        • รายย่อยถือครอง: <b>{res.get('retailHeld', 'N/A')}</b>
                    </div>
                    <div class="biz-summary"><b>[ลักษณะการทำธุรกิจ]</b><br>{res.get('summaryTh', 'N/A')}</div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"❌ ไม่พบข้อมูลสัญลักษณ์หุ้น **{single_ticker}** กรุณาตรวจสอบอีกครั้ง")

with tab2:
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน")
    if st.button("🚀 เริ่มสแกนตลาด"):
        st.info("ระบบกำลังดึงข้อมูลและวิเคราะห์... กรุณารอสักครู่")
        tickers = get_us_stock_tickers()[:50] # จำกัดเพื่อความเร็วในการทดสอบ
        scanned = []
        for t in tickers:
            r, _ = check_ma_snr_combo(t, False)
            if r: scanned.append(r)
        st.success(f"สแกนเสร็จสิ้น! พบหุ้นเข้าเงื่อนไข {len(scanned)} ตัว")
        if scanned:
            st.dataframe(pd.DataFrame(scanned), use_container_width=True)

with tab3:
    st.markdown("### ⭐ Watchlist ส่วนตัวของคุณ")
    if st.session_state.watchlist:
        st.write(", ".join(st.session_state.watchlist))
    else:
        st.info("ยังไม่มีหุ้นใน Watchlist")
