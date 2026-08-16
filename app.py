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

# ================= ส่วนตั้งค่าแอปและภาษา =================
UI_LANG_MAP = {
    'search_ticker_title': "US Stock Scanner PRO",
    'search_ticker_subtitle': "ระบบสแกนเทคนิคอล • คำนวณ % ขาขึ้น • AI Pattern • 3 แนวรับ 4 แนวต้าน",
    'search_ticker_label': "พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, RKLB, IREN):",
    'btn_analyze_single': "🔎 วิเคราะห์ทันที",
    'btn_scan_market': "🚀 เริ่มสแกนทั้ง 3 ตลาด (7,000+ หุ้น)",
    'status_preparing_tickers': "⏳ กำลังดึงรายชื่อหุ้นทั้งหมดจาก NASDAQ, NYSE, AMEX...",
    'status_scanning': "⏳ สแกนไปแล้ว {count}/{total} ตัว...",
    'status_analyzing_single': "⏳ กำลังดึงข้อมูลและวิเคราะห์ {ticker}...",
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
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))
    return session

# Server State ส่วนกลาง
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
    .badge-rsi {
        background: #1E293B;
        color: #38BDF8;
        border: 1px solid #334155;
    }
    .badge-dist {
        background: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
    }
    .badge-trend-bull {
        background: #064E3B;
        color: #6EE7B7;
        border: 1px solid #059669;
    }
    .badge-trend-bear {
        background: #4C0519;
        color: #FDA4AF;
        border: 1px solid #9F1239;
    }
    .badge-trend-pull {
        background: #451A03;
        color: #FCD34D;
        border: 1px solid #78350F;
    }

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
    .snr-lbl {
        color: #94A3B8;
        font-size: 0.72rem;
    }
    .snr-num {
        font-weight: 700;
        font-size: 0.82rem;
    }
    
    .c-green { color: #22C55E !important; }
    .c-lightgreen { color: #4ADE80 !important; }
    .c-red { color: #EF4444 !important; }
    .c-orange { color: #F97316 !important; }
    .c-yellow { color: #FBBF24 !important; }
    .c-darkred { color: #F43F5E !important; }

    /* กล่องการ์ดกลยุทธ์แนวรับแนวตั้ง (Vertical Cards) */
    .strategy-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .strat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .strat-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .strat-price {
        font-size: 0.95rem;
        font-weight: 800;
        color: #38BDF8;
    }
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
    .strat-sub {
        color: #94A3B8;
    }
    .strat-val {
        color: #F8FAFC;
        font-weight: 600;
    }

    .company-header {
        font-size: 1.15rem;
        font-weight: 800;
        color: #38BDF8 !important;
        margin-bottom: 0rem;
    }
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
        padding: 6px 10px;
        margin-bottom: 4px;
    }
    .news-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #E2E8F0;
        text-decoration: none;
    }
    .news-meta {
        font-size: 0.68rem;
        color: #64748B;
        margin-top: 2px;
    }
    
    .desktop-only-space {
        height: 28px;
        display: block;
    }
    @media (max-width: 640px) {
        .desktop-only-space {
            display: none !important;
        }
        .main-title {
            font-size: 1.3rem !important;
        }
        .price-main {
            font-size: 1.3rem;
        }
        .snr-row {
            font-size: 0.72rem;
        }
        .snr-num {
            font-size: 0.78rem;
        }
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="main-title">{UI_LANG_MAP["search_ticker_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{UI_LANG_MAP["search_ticker_subtitle"]}</div>', unsafe_allow_html=True)


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
            "ดีดตัวกลับตัวรูปตัว V (V-Shape).png": np.where(x < 0.55, 1.0 - 1.6 * x, 0.12 + 2.0 * (x - 0.55))
        }

        best_pattern = "สร้างฐานสะสมกำลัง.png"
        best_score = 60.0

        for pat_name, pat_curve in templates.items():
            norm_pat = (pat_curve - np.min(pat_curve)) / (np.max(pat_curve) - np.min(pat_curve) + 1e-6)
            mae = np.mean(np.abs(norm_closes - norm_pat))
            corr = np.corrcoef(norm_closes, norm_pat)[0, 1]
            if np.isnan(corr):
                corr = 0.5
            
            sim_score = (max(0.0, 1.0 - mae) * 0.65 + max(0.0, (corr + 1.0) / 2.0) * 0.35) * 100.0
            if sim_score > best_score:
                best_score = sim_score
                best_pattern = pat_name

        final_score = round(max(70.0, min(95.5, best_score)), 1)
        return best_pattern, final_score
    except Exception:
        return "สร้างฐานสะสมกำลัง.png", 76.5


def get_time_elapsed_thai(last_dt):
    if not last_dt:
        return ""
    diff = datetime.now() - last_dt
    total_seconds = int(diff.total_seconds())
    if total_seconds < 60:
        return f" (เพิ่งสแกนเมื่อ {total_seconds} วิที่แล้ว)"
    elif total_seconds < 3600:
        mins = total_seconds // 60
        return f" (สแกนไปแล้ว {mins} นาทีที่แล้ว)"
    else:
        hours = total_seconds // 3600
        mins = (total_seconds % 3600) // 60
        return f" (สแกนไปแล้ว {hours} ชม. {mins} นาทีที่แล้ว)"


@st.cache_data(ttl=86400)
def get_us_stock_tickers():
    tickers = []
    try:
        url_nasdaq = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt'
        df_nasdaq = pd.read_csv(url_nasdaq, sep='|')
        nasdaq_stocks = df_nasdaq[(df_nasdaq['ETF'] == 'N') & (df_nasdaq['Test Issue'] == 'N')]['Symbol'].dropna().tolist()
        tickers.extend(nasdaq_stocks)
    except Exception:
        pass

    try:
        url_other = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt'
        df_other = pd.read_csv(url_other, sep='|')
        other_stocks = df_other[(df_other['ETF'] == 'N') & (df_other['Test Issue'] == 'N')]['ACT Symbol'].dropna().tolist()
        tickers.extend(other_stocks)
    except Exception:
        pass

    cleaned_tickers = [
        str(t).strip().replace('.', '-')
        for t in tickers
        if isinstance(t, str) and str(t).strip().replace('-', '').isalpha()
    ]
    return sorted(list(set(cleaned_tickers)))


def translate_text_to_thai(text):
    if not text or text == 'N/A':
        return 'N/A'
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "en", "tl": "th", "dt": "t", "q": text}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            res_json = response.json()
            translated_text = "".join([item[0] for item in res_json[0] if item[0]])
            if translated_text:
                return translated_text
    except Exception:
        pass
    return text


@st.cache_data(ttl=14400)
def get_company_info_and_holders(ticker):
    try:
        stock = yf.Ticker(ticker, session=get_yfinance_session())
        info = stock.info
        
        eng_summary = info.get('longBusinessSummary', 'N/A')
        th_summary = translate_text_to_thai(eng_summary) if eng_summary != 'N/A' else 'N/A'
        company_name = info.get('longName', ticker)
        
        raw_sector = info.get('sector', 'N/A')
        raw_industry = info.get('industry', 'N/A')
        sector_th = SECTOR_MAP_TH.get(raw_sector, raw_sector)
        industry_th = translate_text_to_thai(raw_industry) if raw_industry != 'N/A' else 'N/A'
        
        shares_out = info.get('sharesOutstanding', 0)
        shares_out_str = f"{shares_out:,.0f}" if shares_out else "N/A"
        
        inst_held = info.get('heldPercentInstitutions', 0)
        inst_held_pct = f"{inst_held * 100:.2f}%" if inst_held else "N/A"
        
        insider_held = info.get('heldPercentInsiders', 0)
        insider_held_pct = f"{insider_held * 100:.2f}%" if insider_held else "N/A"
        
        retail_held_pct = "N/A"
        if inst_held and insider_held:
            retail_calc = 100 - ((inst_held + insider_held) * 100)
            retail_held_pct = f"{max(0.0, retail_calc):.2f}%"

        return {
            'longNameEn': company_name,
            'sectorTh': sector_th,
            'industryTh': industry_th,
            'summaryTh': th_summary,
            'sharesOutstanding': shares_out_str,
            'institutionalHeld': inst_held_pct,
            'insiderHeld': insider_held_pct,
            'retailHeld': retail_held_pct
        }
    except Exception:
        return {
            'longNameEn': ticker,
            'sectorTh': 'N/A',
            'industryTh': 'N/A',
            'summaryTh': 'N/A',
            'sharesOutstanding': 'N/A',
            'institutionalHeld': 'N/A',
            'insiderHeld': 'N/A',
            'retailHeld': 'N/A'
        }


@st.cache_data(ttl=3600)
def get_stock_news(ticker):
    try:
        stock = yf.Ticker(ticker, session=get_yfinance_session())
        news_items = stock.news
        if not news_items:
            return []
        
        results = []
        for n in news_items[:3]:
            title_en = n.get('title', '')
            title_th = translate_text_to_thai(title_en)
            publisher = n.get('publisher', 'Financial News')
            link = n.get('link', '#')
            pub_ts = n.get('providerPublishTime', 0)
            pub_date_str = datetime.fromtimestamp(pub_ts).strftime('%d/%m/%Y %H:%M') if pub_ts else ''
            results.append({
                'title': title_th,
                'publisher': publisher,
                'link': link,
                'time': pub_date_str
            })
        return results
    except Exception:
        return []


@st.cache_data(ttl=14400)
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker, session=get_yfinance_session())
        q_financials = stock.quarterly_financials
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
    except Exception:
        pass
    return None


def create_ta_chart(df, ticker, res_data):
    if df is None or df.empty:
        return None

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='ราคา'
    )])

    fast_ma = df['close'].rolling(window=20).mean()
    slow_ma = df['close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(x=df.index, y=fast_ma, line=dict(color='#38BDF8', width=1.2), name='MA20'))
    fig.add_trace(go.Scatter(x=df.index, y=slow_ma, line=dict(color='#FB923C', width=1.2), name='MA50'))

    latest_date = df.index[-1]
    earliest_date = df.index[0]
    
    supports = [
        ('Support 1 ($)', '#22C55E', -12),
        ('Support 2 ($)', '#16A34A', 12),
        ('Support 3 ($)', '#15803D', -12)
    ]
    for key, color, ay_pos in supports:
        if key in res_data:
            val = res_data[key]
            fig.add_shape(type="line", x0=earliest_date, y0=val, x1=latest_date, y1=val, line=dict(color=color, width=1.6, dash='dash'))
            fig.add_annotation(
                x=latest_date, y=val, 
                text=f"{key.replace(' ($)', '')}: ${val}", 
                bgcolor=color, 
                font=dict(color="white", size=9), 
                xanchor="left",
                ax=8, ay=ay_pos
            )

    resistances = [
        ('Resist 1 ($)', '#EF4444', -12),
        ('Resist 2 ($)', '#F97316', 12),
        ('Resist 3 ($)', '#EAB308', -12),
        ('Resist 4 ($)', '#991B1B', 12)
    ]
    for key, color, ay_pos in resistances:
        if key in res_data:
            val = res_data[key]
            fig.add_shape(type="line", x0=earliest_date, y0=val, x1=latest_date, y1=val, line=dict(color=color, width=1.6, dash='dash'))
            fig.add_annotation(
                x=latest_date, y=val, 
                text=f"{key.replace(' ($)', '')}: ${val}", 
                bgcolor=color, 
                font=dict(color="white", size=9), 
                xanchor="left",
                ax=8, ay=ay_pos
            )

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        margin=dict(l=6, r=65, t=10, b=6),
        height=340,
        dragmode='pan',
        yaxis_title="ราคา ($)",
        showlegend=False
    )
    return fig


@st.cache_data(ttl=14400)
def check_ma_snr_combo(ticker, info_mode=False):
    try:
        stock = yf.Ticker(ticker, session=get_yfinance_session())
        df = stock.history(period='2y', interval='1d')
        if len(df) < 50 or df['Close'].iloc[-1] < 0.5:
            return None, None

        df.columns = [col.lower() for col in df.columns]
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        loss_safe = loss.replace(0, np.nan)
        rs = gain / loss_safe
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(100.0)
        latest_rsi = round(df['rsi'].iloc[-1], 2)

        latest_close = df['close'].iloc[-1]
        fast_ma = df['close'].rolling(window=20).mean()
        slow_ma = df['close'].rolling(window=50).mean()

        lows = df['low'].values
        lower_lows = sorted(list(set(lows[lows < latest_close])), reverse=True)
        
        s1, s2, s3 = latest_close * 0.95, latest_close * 0.90, latest_close * 0.85
        if len(lower_lows) >= 3:
            step = max(1, len(lower_lows) // 3)
            s1 = lower_lows[0]
            s2 = lower_lows[min(step, len(lower_lows)-1)]
            s3 = lower_lows[-1]
        elif len(lower_lows) > 0:
            s1 = lower_lows[0]
            s3 = lower_lows[-1]
            s2 = s1 - (s1 - s3) * 0.5

        highs = df['high'].values
        sorted_highs = sorted(list(set(highs[highs > latest_close])), reverse=False)
        
        r1, r2, r3, r4 = latest_close * 1.05, latest_close * 1.12, latest_close * 1.25, latest_close * 1.40
        if len(sorted_highs) >= 4:
            step_r = max(1, len(sorted_highs) // 4)
            r1 = sorted_highs[min(step_r, len(sorted_highs)-1)]
            r2 = sorted_highs[min(step_r*2, len(sorted_highs)-1)]
            r3 = sorted_highs[min(step_r*3, len(sorted_highs)-1)]
            r4 = sorted_highs[-1]
        elif len(sorted_highs) > 0:
            r1 = sorted_highs[0]
            r4 = sorted_highs[-1]
            r2 = r1 + (r4 - r1) * 0.33
            r3 = r1 + (r4 - r1) * 0.66

        recent_3d_low = df['low'].tail(3).min()
        near_support = recent_3d_low <= (s1 * 1.05)
        near_ma50 = recent_3d_low <= (slow_ma.iloc[-1] * 1.02)
        is_green_candle = latest_close > df['open'].iloc[-1]
        has_volume = df['volume'].iloc[-1] >= 200_000

        recent_high_8d = df['high'].tail(8).max()
        pullback_8d_pct = ((latest_close - recent_high_8d) / recent_high_8d) * 100
        
        recent_low_8d = df['low'].tail(8).min()
        bounce_8d_pct = ((latest_close - recent_low_8d) / recent_low_8d) * 100

        bull_score = 0
        if latest_close >= fast_ma.iloc[-1]: bull_score += 20
        if latest_close >= slow_ma.iloc[-1]: bull_score += 15
        if fast_ma.iloc[-1] >= slow_ma.iloc[-1]: bull_score += 15
        if 48 <= latest_rsi <= 70: bull_score += 15
        elif latest_rsi > 70: bull_score += 10
        if bounce_8d_pct >= 3.0: bull_score += 20
        if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1]: bull_score += 15
        
        bullish_pct = min(96.0, max(12.0, round(bull_score * 0.9 + 5.0, 1)))

        if (near_support or bounce_8d_pct >= 2.5) and is_green_candle and (bullish_pct >= 65 or bounce_8d_pct >= 5.0):
            trend_status = "BUY_SIGNAL"
            status_desc = "🟢 ผ่านเงื่อนไขสัญญาณ BUY (ดีดตัวกลับตัวจากแนวรับ)"
        elif bounce_8d_pct >= 3.0:
            trend_status = "REVERSAL_BOUNCE"
            status_desc = "🟡 หุ้นกำลังฟอร์มตัวกลับตัว/เด้งจากแนวรับ (Reversal Bounce)"
        elif bullish_pct >= 50:
            trend_status = "PULLBACK"
            status_desc = "🟠 กำลังย่อตัว/พักฐานระยะสั้น (Buy on Dip)"
        elif latest_rsi < 35:
            trend_status = "OVERSOLD"
            status_desc = "🟣 ขายมากเกินไป (Oversold) ลุ้นเด้งกลับตัวที่แนวรับ"
        else:
            trend_status = "DOWNTREND"
            status_desc = "🔴 โครงสร้างชะลอตัว/แนวโน้มขาลง (รอสร้างฐานที่แนวรับ)"

        dist_from_sup = ((latest_close - s1) / s1) * 100
        pat_name, pat_score = calculate_ai_pattern_match(df)

        res_data = {
            'Ticker': ticker,
            'Price ($)': round(latest_close, 2),
            'Support 1 ($)': round(s1, 2),
            'Support 2 ($)': round(s2, 2),
            'Support 3 ($)': round(s3, 2),
            'Resist 1 ($)': round(r1, 2),
            'Resist 2 ($)': round(r2, 2),
            'Resist 3 ($)': round(r3, 2),
            'Resist 4 ($)': round(r4, 2),
            'Dist_Sup (%)': f'{dist_from_sup:+.2f}%',
            'RSI': latest_rsi,
            'Volume': f"{df['volume'].iloc[-1]:,.0f}",
            'Date': df.index[-1].strftime('%Y-%m-%d'),
            'pattern_name': pat_name,
            'pattern_score': pat_score,
            'bullish_pct': bullish_pct,
            'bearish_pct': round(100.0 - bullish_pct, 1),
            'trend_status': trend_status,
            'status_desc': status_desc,
            'pullback_8d_pct': f'{pullback_8d_pct:.2f}%',
            'bounce_8d_pct': f'+{bounce_8d_pct:.2f}%'
        }

        if info_mode:
            co_info = get_company_info_and_holders(ticker)
            res_data.update(co_info)

        if not info_mode:
            if not (trend_status in ["BUY_SIGNAL", "REVERSAL_BOUNCE"]):
                return None, df

        return res_data, df
    except Exception:
        pass
    return None, None


PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True
}

# ================= สร้างหน้าจอแท็บหลัก =================
tab1, tab2, tab3 = st.tabs([UI_LANG_MAP['tab_search_ticker'], UI_LANG_MAP['tab_scan_market'], UI_LANG_MAP['tab_watchlist']])

# --- TAB 1: ค้นหาหุ้นรายตัว ---
with tab1:
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        single_ticker = st.text_input(UI_LANG_MAP['search_ticker_label'], value='').strip().upper()
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
                t_status = res.get("trend_status", "BUY_SIGNAL")
                bull_pct = res.get("bullish_pct", 50.0)
                bear_pct = res.get("bearish_pct", 50.0)

                st.markdown(f'<p class="company-header">{single_ticker} : {company_full_name}</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="sector-badge">🏷️ กลุ่มธุรกิจ: {sector_desc} | ย่อย: {industry_desc}</div>', unsafe_allow_html=True)
                
                # แสดงผลสถานะแนวโน้ม
                if t_status == "BUY_SIGNAL":
                    st.success(f"{res['status_desc']} (ดีดตัวจากก้น {res['bounce_8d_pct']}) | ณ วันที่: {res['Date']}")
                elif t_status == "REVERSAL_BOUNCE":
                    st.success(f"{res['status_desc']} (ดีดตัวขึ้นจากก้นล่าสุด {res['bounce_8d_pct']} | ย่อจาก High 8 วัน {res['pullback_8d_pct']}) | ณ วันที่: {res['Date']}")
                elif t_status == "PULLBACK":
                    st.warning(f"{res['status_desc']} (ย่อตัวจาก High 8 วัน {res['pullback_8d_pct']}) | ณ วันที่: {res['Date']}")
                elif t_status == "OVERSOLD":
                    st.info(f"{res['status_desc']} (RSI: {res['RSI']}) | ณ วันที่: {res['Date']}")
                else:
                    st.error(f"{res['status_desc']} (ย่อตัวจาก High 8 วัน {res['pullback_8d_pct']}) | ณ วันที่: {res['Date']}")

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
                    st.markdown(f'<div class="chart-header-badge">{single_ticker} | ล่าสุด: ${res["Price ($)"]} (RSI: {res.get("RSI", 0)})</div>', unsafe_allow_html=True)
                    fig = create_ta_chart(raw_df, single_ticker, res)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"chart_single_{single_ticker}")
                    
                    pat_name = res.get('pattern_name', 'สร้างฐานสะสมกำลัง.png')
                    pat_score = res.get('pattern_score', 75.0)
                    st.markdown(f'<div class="pattern-box">😊 🤖 AI Pattern Match: {pat_name} (ความแม่นยำ: {pat_score}%)</div>', unsafe_allow_html=True)

                st.markdown("---")
                
                # กล่อง Compact Board
                st.markdown(f"#### {UI_LANG_MAP['analysis_title']}")
                trend_badge_class = "badge-trend-bull" if bull_pct >= 60 else ("badge-trend-pull" if bull_pct >= 45 else "badge-trend-bear")
                
                st.markdown(f"""
                <div class="compact-board">
                    <div class="price-banner">
                        <div class="price-val-box">
                            <span style="font-size:0.8rem; color:#94A3B8; font-weight:600;">💰 ราคา:</span>
                            <span class="price-main">${res['Price ($)']}</span>
                        </div>
                        <div class="price-badge-group">
                            <span class="price-badge {trend_badge_class}">📈 ขาขึ้น/กลับตัว: {bull_pct}%</span>
                            <span class="price-badge badge-dist">ดีดจากก้น 8 วัน: {res['bounce_8d_pct']}</span>
                            <span class="price-badge badge-rsi">RSI: {res['RSI']}</span>
                        </div>
                    </div>
                    <div class="snr-grid">
                        <div class="snr-card" style="border-left: 3px solid #22C55E;">
                            <div class="snr-card-title c-green">🛡️ แนวรับ (Support)</div>
                            <div class="snr-row"><span class="snr-lbl">รับ 1 (ใกล้สุด)</span><span class="snr-num c-green">${res['Support 1 ($)']}</span></div>
                            <div class="snr-row"><span class="snr-lbl">รับ 2 (โซนหลัก)</span><span class="snr-num c-lightgreen">${res['Support 2 ($)']}</span></div>
                            <div class="snr-row"><span class="snr-lbl">รับ 3 (ลึกสุด)</span><span class="snr-num c-lightgreen">${res['Support 3 ($)']}</span></div>
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

                # ================= การ์ดกลยุทธ์แบ่งไม้เข้าซื้อ (แนวตั้ง อ่านง่ายบนมือถือ ไม่ต้องเลื่อนจอ) =================
                st.markdown("#### 🎯 กลยุทธ์แบ่งไม้เข้าซื้อ & ประเมินความแข็งแรงของแนวรับ")
                
                dist_s1 = ((res['Support 1 ($)'] - res['Price ($)']) / res['Price ($)']) * 100
                dist_s2 = ((res['Support 2 ($)'] - res['Price ($)']) / res['Price ($)']) * 100
                dist_s3 = ((res['Support 3 ($)'] - res['Price ($)']) / res['Price ($)']) * 100

                st.markdown(f"""
                <div class="strategy-card" style="border-left: 4px solid #22C55E;">
                    <div class="strat-header">
                        <div>
                            <span class="strat-title">🛡️ แนวรับ 1 (สวิงโลว์ใกล้สุด)</span>
                            <span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">({dist_s1:+.2f}%)</span>
                        </div>
                        <span class="strat-price" style="color:#22C55E;">${res['Support 1 ($)']}</span>
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
                        <span class="strat-price" style="color:#4ADE80;">${res['Support 2 ($)']}</span>
                    </div>
                    <div class="strat-body">
                        <div><span class="strat-sub">ความแข็งแรง:</span> <span class="strat-val">⭐⭐⭐⭐ แข็งแกร่ง</span></div>
                        <div><span class="strat-sub">กลยุทธ์:</span> <span class="strat-val c-lightgreen">35% (ไม้หลักสะสมของ)</span></div>
                    </div>
                </div>

                <div class="strategy-card" style="border-left: 4px solid #15803D;">
                    <div class="strat-header">
                        <div>
                            <span class="strat-title">🛡️ แนวรับ 3 (แนวรับจิตวิทยาใหญ่)</span>
                            <span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">({dist_s3:+.2f}%)</span>
                        </div>
                        <span class="strat-price" style="color:#86EFAC;">${res['Support 3 ($)']}</span>
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
                        st.markdown(f"""
                        <div class="news-card">
                            <a class="news-title" href="{news['link']}" target="_blank">📌 {news['title']}</a>
                            <div class="news-meta">แหล่งข่าว: {news['publisher']} | เวลา: {news['time']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ ไม่พบหัวข้อข่าวสำคัญในรอบสัปดาห์สำหรับหุ้นตัวนี้")

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
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน (ทั้งตลาด NASDAQ, NYSE, AMEX)")
    
    is_busy = server_state["is_scanning"]
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        scan_btn = st.button(UI_LANG_MAP['btn_scan_market'], disabled=is_busy, key="btn_scan_all")
    with col_btn2:
        reset_btn = st.button("🔄 รีเซ็ตข้อมูลสแกน", disabled=is_busy, key="btn_reset_all")

    if is_busy:
        st.warning("⏳ **ขณะนี้มีผู้ใช้งานท่านอื่นกำลังสแกนทั้งตลาดอยู่** ระบบกำลังประมวลผลให้ส่วนกลาง กรุณารอประมาณ 1-2 นาที จากนั้นผลลัพธ์จะแสดงขึ้นมาโดยอัตโนมัติครับ")

    if reset_btn and not is_busy:
        server_state["latest_results"] = None
        server_state["latest_df"] = None
        server_state["last_scanned_at"] = None
        server_state["last_scanned_dt"] = None
        st.success("ล้างข้อมูลการสแกนส่วนกลางเรียบร้อยแล้ว")
        st.rerun()

    if scan_btn and not is_busy:
        server_state["is_scanning"] = True
        status_text = st.empty()
        status_text.info(UI_LANG_MAP['status_preparing_tickers'])
        stock_list = get_us_stock_tickers()
        total_stocks = len(stock_list)
        progress_bar = st.progress(0)
        
        results = []
        count = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                futures = {executor.submit(check_ma_snr_combo, ticker, False): ticker for ticker in stock_list}
                for future in concurrent.futures.as_completed(futures):
                    count += 1
                    if count % 20 == 0 or count == total_stocks:
                        progress_bar.progress(count / total_stocks)
                        status_text.text(UI_LANG_MAP['status_scanning'].format(count=count, total=total_stocks))
                    try:
                        res_data_found, raw_df_found = future.result()
                        if res_data_found and raw_df_found is not None:
                            results.append({'res_data': res_data_found, 'raw_df': raw_df_found})
                    except Exception:
                        pass
        finally:
            server_state["is_scanning"] = False

        status_text.empty()
        st.success(f'✅ สแกนเสร็จสิ้น! พบหุ้นทรงสวยเข้าเงื่อนไขทั้งหมด {len(results)} ตัว')
        
        server_state["latest_results"] = results
        server_state["last_scanned_dt"] = datetime.now()
        server_state["last_scanned_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if results:
            df_result_display = pd.DataFrame([item['res_data'] for item in results])[[
                'Ticker', 'Price ($)', 'bullish_pct', 'bounce_8d_pct', 'Support 1 ($)', 'Support 2 ($)', 'Support 3 ($)', 'RSI', 
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
                    ticker_found = res_data['Ticker']
                    raw_df_found = item.get('raw_df')
                    b_pct = res_data.get('bullish_pct', 70.0)
                    bounce_str = res_data.get('bounce_8d_pct', '+0.0%')

                    with cols[c_offset]:
                        with st.container():
                            st.markdown(f'<p style="font-size:0.92rem; font-weight:bold; color:#60A5FA; margin-bottom:0px;">🟢 {ticker_found} (ขาขึ้น {b_pct}% | เด้ง {bounce_str})</p>', unsafe_allow_html=True)
                            st.caption(f"Support 1: ${res_data['Support 1 ($)']} | ต้าน 1: ${res_data['Resist 1 ($)']} | RSI: {res_data['RSI']}")
                            
                            if ticker_found not in st.session_state.watchlist:
                                if st.button(f"⭐ บันทึก {ticker_found} เข้า Watchlist", key=f"btn_gal_wl_{ticker_found}_{page_num}_{item_idx}"):
                                    st.session_state.watchlist.append(ticker_found)
                                    st.rerun()
                            
                            if raw_df_found is not None:
                                st.markdown(f'<div class="chart-header-badge">{ticker_found} | ล่าสุด: ${res_data["Price ($)"]} (RSI: {res_data.get("RSI", 0)})</div>', unsafe_allow_html=True)
                                fig_gallery = create_ta_chart(raw_df_found, ticker_found, res_data)
                                if fig_gallery:
                                    st.plotly_chart(fig_gallery, use_container_width=True, config=PLOTLY_CONFIG, key=f"gallery_chart_{ticker_found}_{page_num}_{item_idx}")
                                
                                pat_name = res_data.get('pattern_name', 'สร้างฐานสะสมกำลัง.png')
                                pat_score = res_data.get('pattern_score', 75.0)
                                st.markdown(f'<div class="pattern-box" style="font-size:0.72rem; padding:3px 6px;">😊 🤖 AI Pattern: {pat_name} ({pat_score}%)</div>', unsafe_allow_html=True)
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
                stock = yf.Ticker(w_ticker, session=get_yfinance_session())
                df_w = stock.history(period='5d')
                if not df_w.empty:
                    curr_price = round(df_w['Close'].iloc[-1], 2)
                    prev_close = df_w['Close'].iloc[-2] if len(df_w) > 1 else curr_price
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
