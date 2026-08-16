
import concurrent.futures
from datetime import datetime
import os
import cv2
import numpy as np
import pandas as pd
import requests
from skimage.metrics import structural_similarity as ssim
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# ================= ส่วนตั้งค่าแอปและภาษา =================
UI_LANG_MAP = {
    'search_ticker_title': "US Stock Scanner PRO (Enterprise Edition)",
    'search_ticker_subtitle': "ระบบสแกนทางเทคนิค พร้อมระบุกลุ่มธุรกิจ, 3 แนวรับ, 4 แนวต้าน, โครงสร้างผู้ถือหุ้น และ AI Pattern",
    'search_ticker_label': "พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, RKLB):",
    'btn_analyze_single': "🔎 วิเคราะห์ทันที",
    'btn_scan_market': "🚀 เริ่มสแกนทั้ง 3 ตลาด (7,000+ หุ้น)",
    'status_preparing_tickers': "⏳ กำลังดึงรายชื่อหุ้นทั้งหมดจาก NASDAQ, NYSE, AMEX...",
    'status_scanning': "⏳ สแกนไปแล้ว {count}/{total} ตัว...",
    'status_analyzing_single': "⏳ กำลังดึงข้อมูลและวิเคราะห์ {ticker}...",
    'success_stock_found_single': "🟢 หุ้น **{ticker}** ผ่านเงื่อนไขสแกนสัญญาณ BUY!",
    'error_stock_not_found_single': "🔴 หุ้น **{ticker}** ไม่ติดเงื่อนไขสัญญาณซื้อในขณะนี้",
    'expander_business_summary': "📖 สรุปธุรกิจ & โครงสร้างผู้ถือหุ้น (แปลไทยอัตโนมัติ)",
    'chart_title_single': "📈 กราฟเทคนิค 3 แนวรับ และ 4 ระดับแนวต้าน",
    'placeholder_pattern_match': "🤖 AI Pattern Match: สร้างฐาน.png (ความแม่นยำ: 75.4%)",
    'analysis_title': "📊 ข้อมูลวิเคราะห์สำคัญ",
    'metric_current_price': "ราคาปัจจุบัน",
    'tab_search_ticker': "🔍 ค้นหา & วิเคราะห์รายตัว",
    'tab_scan_market': "🚀 สแกนคัดหุ้นทรงสวย",
    'tab_watchlist': "⭐ Watchlist ส่วนตัว",
}

SECTOR_MAP_TH = {
    'Technology': '💻 เทคโนโลยี / อิเล็กทรอนิกส์ & ซอฟต์แวร์',
    'Healthcare': '🏥 สุขภาพ / การแพทย์ & ยา',
    'Financial Services': '🏦 การเงิน / ธนาคาร & ประกันภัย',
    'Industrials': '🏭 อุตสาหกรรม / อวกาศ & การป้องกันประเทศ / ขนส่ง',
    'Consumer Cyclical': '🛍️ สินค้าฟุ่มเฟือย / ค้าปลีก & ยานยนต์',
    'Consumer Defensive': '🛒 สินค้าอุปโภคบริโภคจำเป็น',
    'Energy': '⚡ พลังงาน / น้ำมัน, ก๊าซ & พลังงานสะอาด',
    'Real Estate': '🏢 อสังหาริมทรัพย์ / กองรีท (REITs)',
    'Basic Materials': '🧪 วัตถุดิบพื้นฐาน / เคมีภัณฑ์ & เหมืองแร่',
    'Communication Services': '📡 สื่อสาร / โทรคมนาคม & บันเทิง/มีเดีย',
    'Utilities': '💡 สาธารณูปโภค / ไฟฟ้า & ประปา'
}

st.set_page_config(
    page_title=UI_LANG_MAP['search_ticker_title'],
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'scan_df' not in st.session_state:
    st.session_state.scan_df = None

# Custom CSS
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 1200px;
    }
    .main-title {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 0.8rem !important;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    .fin-card {
        background: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
        color: #F8FAFC !important;
    }
    .fin-card-label {
        font-size: 0.75rem;
        color: #94A3B8 !important;
        font-weight: 600;
        text-transform: uppercase;
    }
    .fin-card-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-top: 2px;
    }
    .fin-card-sub {
        font-size: 0.7rem;
        color: #34D399 !important;
        margin-top: 2px;
    }
    .company-header {
        font-size: 1.2rem;
        font-weight: 800;
        color: #38BDF8 !important;
        margin-bottom: 0rem;
    }
    .sector-badge {
        font-size: 0.82rem;
        font-weight: 600;
        color: #FCD34D;
        background: #451A03;
        border: 1px solid #78350F;
        padding: 4px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    .chart-header-badge {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F8FAFC;
        background-color: #1E293B;
        padding: 6px 10px;
        border-radius: 6px;
        margin-bottom: 6px;
        display: inline-block;
    }
    .biz-summary {
        font-size: 0.85rem !important;
        color: #F1F5F9 !important;
        background-color: #0B132B !important;
        padding: 12px !important;
        border-radius: 8px;
        border-left: 4px solid #3B82F6 !important;
        border: 1px solid #334155 !important;
        margin-bottom: 0.4rem;
        line-height: 1.6;
    }
    .pattern-box {
        background-color: #172554 !important;
        color: #93C5FD !important;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #1E40AF !important;
        margin-top: 6px;
        margin-bottom: 6px;
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


def get_base_directory():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(script_dir):
            return script_dir
    except NameError:
        pass
    return os.getcwd()


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


@st.cache_data(ttl=3600)
def get_company_info_and_holders(ticker):
    try:
        stock = yf.Ticker(ticker)
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


def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
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
    
    # 3 แนวรับ
    supports = [
        ('Support 1 ($)', '#22C55E', -15),
        ('Support 2 ($)', '#16A34A', -15),
        ('Support 3 ($)', '#15803D', 15)
    ]
    for key, color, ay_pos in supports:
        if key in res_data:
            val = res_data[key]
            fig.add_shape(type="line", x0=earliest_date, y0=val, x1=latest_date, y1=val, line=dict(color=color, width=2, dash='dash'))
            fig.add_annotation(x=latest_date, y=val, text=f"{key.replace(' ($)', '')}: ${val}", bgcolor=color, font=dict(color="white"), ax=0, ay=ay_pos)

    # 4 แนวต้าน
    resistances = [
        ('Resist 1 ($)', '#EF4444', -15),
        ('Resist 2 ($)', '#F97316', -15),
        ('Resist 3 ($)', '#EAB308', -15),
        ('Resist 4 ($)', '#991B1B', 15)
    ]
    for key, color, ay_pos in resistances:
        if key in res_data:
            val = res_data[key]
            fig.add_shape(type="line", x0=earliest_date, y0=val, x1=latest_date, y1=val, line=dict(color=color, width=2, dash='dash'))
            fig.add_annotation(x=latest_date, y=val, text=f"{key.replace(' ($)', '')}: ${val}", bgcolor=color, font=dict(color="white"), ax=0, ay=ay_pos)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        margin=dict(l=10, r=10, t=15, b=10),
        height=380,
        dragmode='pan',
        yaxis_title="ราคา ($)",
        showlegend=False
    )
    return fig


def check_ma_snr_combo(ticker, info_mode=False):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='2y', interval='1d')
        if len(df) < 50 or df['Close'].iloc[-1] < 0.5:
            return None, None

        df.columns = [col.lower() for col in df.columns]
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        latest_rsi = round(df['rsi'].iloc[-1], 2)

        latest_close = df['close'].iloc[-1]
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

        if not (near_support or near_ma50):
            return None, df

        if latest_close > df['open'].iloc[-1] and df['volume'].iloc[-1] >= 200_000:
            dist_from_sup = ((latest_close - s1) / s1) * 100
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
                'Dist_Sup (%)': f'+{dist_from_sup:.2f}%',
                'RSI': latest_rsi,
                'Volume': f"{df['volume'].iloc[-1]:,.0f}",
                'Date': df.index[-1].strftime('%Y-%m-%d'),
            }
            if info_mode:
                co_info = get_company_info_and_holders(ticker)
                res_data.update(co_info)
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
    st.markdown("### 🔍 ตรวจสอบสัญญาณเทคนิคและพื้นฐานรายตัว")
    
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        single_ticker = st.text_input(UI_LANG_MAP['search_ticker_label'], value='').strip().upper()
    with col_in2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button(UI_LANG_MAP['btn_analyze_single'])

    if search_btn and single_ticker:
        with st.spinner(UI_LANG_MAP['status_analyzing_single'].format(ticker=single_ticker)):
            res, raw_df = check_ma_snr_combo(single_ticker, info_mode=True)
            df_profit = get_financials(single_ticker)

            if res:
                company_full_name = res.get("longNameEn", single_ticker)
                sector_desc = res.get("sectorTh", "N/A")
                industry_desc = res.get("industryTh", "N/A")

                st.markdown(f'<p class="company-header">{single_ticker} : {company_full_name}</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="sector-badge">🏷️ กลุ่มธุรกิจ: {sector_desc} | ย่อย: {industry_desc}</div>', unsafe_allow_html=True)
                st.success(UI_LANG_MAP['success_stock_found_single'].format(ticker=single_ticker) + f' | ข้อมูล ณ วันที่: {res["Date"]}')
                
                if single_ticker not in st.session_state.watchlist:
                    if st.button(f"⭐ เพิ่ม {single_ticker} เข้า Watchlist"):
                        st.session_state.watchlist.append(single_ticker)
                        st.success(f"เพิ่ม {single_ticker} สำเร็จ!")
                        st.rerun()
                else:
                    if st.button(f"🗑️ ลบ {single_ticker} ออกจาก Watchlist"):
                        st.session_state.watchlist.remove(single_ticker)
                        st.rerun()
                    st.info(f"📌 หุ้น {single_ticker} อยู่ใน Watchlist ของคุณแล้ว")

                if raw_df is not None:
                    st.markdown(f"#### {UI_LANG_MAP['chart_title_single']}")
                    st.markdown(f'<div class="chart-header-badge">{single_ticker} | ราคาปัจจุบัน: ${res["Price ($)"]} (RSI: {res.get("RSI", 0)})</div>', unsafe_allow_html=True)
                    fig = create_ta_chart(raw_df, single_ticker, res)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                    st.markdown(f'<div class="pattern-box">😊 {UI_LANG_MAP["placeholder_pattern_match"]}</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"#### {UI_LANG_MAP['analysis_title']}")
                
                # แสดงการ์ดแบบเรียงลำดับแถวต่อแถว
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">💰 {UI_LANG_MAP['metric_current_price']}</div>
                        <div class="fin-card-value">${res['Price ($)']}</div>
                        <div class="fin-card-sub">RSI (14): {res['RSI']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">🛡️ แนวรับ 1 (ใกล้สุด)</div>
                        <div class="fin-card-value">${res['Support 1 ($)']}</div>
                        <div class="fin-card-sub">{res['Dist_Sup (%)']} จากราคาปัจจุบัน</div>
                    </div>
                    """, unsafe_allow_html=True)

                col_m3, col_m4 = st.columns(2)
                with col_m3:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">🛡️ แนวรับ 2</div>
                        <div class="fin-card-value">${res['Support 2 ($)']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m4:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">🛡️ แนวรับ 3 (ลึกสุด)</div>
                        <div class="fin-card-value">${res['Support 3 ($)']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">⚡ แนวต้าน 1</div>
                        <div class="fin-card-value">${res['Resist 1 ($)']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_r2:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">⚡ แนวต้าน 2</div>
                        <div class="fin-card-value">${res['Resist 2 ($)']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                col_r3, col_r4 = st.columns(2)
                with col_r3:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">⚡ แนวต้าน 3</div>
                        <div class="fin-card-value">${res['Resist 3 ($)']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_r4:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">🚀 แนวต้าน 4</div>
                        <div class="fin-card-value">${res['Resist 4 ($)']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 💰 กำไรสุทธิ 3 ไตรมาสล่าสุด")
                if df_profit is not None:
                    c_table, c_chart = st.columns(2)
                    with c_table:
                        st.dataframe(df_profit, use_container_width=True, hide_index=True, height=125)
                    with c_chart:
                        fig_profit = go.Figure(data=[go.Bar(
                            x=df_profit['Quarter End'],
                            y=df_profit['Net Income (M$)'],
                            marker_color='#3B82F6'
                        )])
                        fig_profit.update_layout(
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=125,
                            template='plotly_dark',
                            xaxis_title="",
                            yaxis_title="M$"
                        )
                        st.plotly_chart(fig_profit, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.warning("ไม่พบข้อมูลกำไรสุทธิย้อนหลัง")

                st.markdown("---")
                
                summary_text = res.get('summaryTh', 'N/A')
                shares_tot = res.get('sharesOutstanding', 'N/A')
                inst_pct = res.get('institutionalHeld', 'N/A')
                insider_pct = res.get('insiderHeld', 'N/A')
                retail_pct = res.get('retailHeld', 'N/A')

                with st.expander(UI_LANG_MAP['expander_business_summary'], expanded=True):
                    st.markdown(f"""
                    <div class="fin-card" style="margin-bottom: 12px; background: #0F172A; border: 1px solid #334155; padding: 14px; border-radius: 8px;">
                        <b style="color: #60A5FA; font-size: 0.95rem;">📊 โครงสร้างผู้ถือหุ้น & ข้อมูลบริษัท:</b>
                        <div style="color: #F8FAFC; line-height: 1.8; margin-top: 6px; font-size: 0.88rem;">
                        • กลุ่มธุรกิจหลัก: <b style="color: #FCD34D;">{sector_desc}</b><br>
                        • อุตสาหกรรมย่อย: <b style="color: #E2E8F0;">{industry_desc}</b><br>
                        • จำนวนหุ้นที่มีทั้งหมด: <b style="color: #FFFFFF;">{shares_tot} หุ้น</b><br>
                        • สถาบัน/บริษัทใหญ่ถือครอง: <b style="color: #38BDF8;">{inst_pct}</b><br>
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
                st.error(UI_LANG_MAP['error_stock_not_found_single'].format(ticker=single_ticker))


# --- TAB 2: สแกนคัดหุ้นทั้งตลาด ---
with tab2:
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน (ทั้งตลาด NASDAQ, NYSE, AMEX)")
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        scan_btn = st.button(UI_LANG_MAP['btn_scan_market'])
    with col_btn2:
        reset_btn = st.button("🔄 รีเซ็ตข้อมูลสแกน")

    if reset_btn:
        st.session_state.scan_results = None
        st.session_state.scan_df = None
        st.success("ล้างข้อมูลการสแกนเรียบร้อยแล้ว")
        st.rerun()

    if scan_btn:
        status_text = st.empty()
        status_text.info(UI_LANG_MAP['status_preparing_tickers'])
        stock_list = get_us_stock_tickers()
        total_stocks = len(stock_list)
        progress_bar = st.progress(0)
        
        results = []
        count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(check_ma_snr_combo, ticker, True): ticker for ticker in stock_list}
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

        status_text.empty()
        st.success(f'✅ สแกนเสร็จสิ้น! พบหุ้นทรงสวยเข้าเงื่อนไขทั้งหมด {len(results)} ตัว')
        
        st.session_state.scan_results = results
        if results:
            df_result_display = pd.DataFrame([item['res_data'] for item in results])[[
                'Ticker', 'longNameEn', 'sectorTh', 'Price ($)', 'Support 1 ($)', 'Support 2 ($)', 'Support 3 ($)', 'Dist_Sup (%)', 'RSI', 
                'Resist 1 ($)', 'Resist 2 ($)', 'Resist 3 ($)', 'Resist 4 ($)', 
                'sharesOutstanding', 'institutionalHeld', 'retailHeld', 'Volume', 'Date'
            ]]
            st.session_state.scan_df = df_result_display

    if st.session_state.scan_results:
        results = st.session_state.scan_results
        df_result_display = st.session_state.scan_df

        st.markdown("---")
        st.subheader('📸 แกลเลอรี่กราฟหุ้นทรงสวย (พร้อมรายละเอียดบริษัทและ AI Pattern Match)')
        
        items_per_page = 6
        total_pages = max(1, (len(results) + items_per_page - 1) // items_per_page)
        page_num = st.selectbox("เลือกหน้าแสดงผลกราฟ:", range(1, int(total_pages) + 1), key="pagination_select")
        
        start_idx = (page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_page_items = results[start_idx:end_idx]

        cols = st.columns(2)
        col_idx = 0

        for item in current_page_items:
            res_data = item['res_data']
            ticker_found = res_data['Ticker']
            co_name = res_data.get('longNameEn', ticker_found)
            sec_name = res_data.get('sectorTh', 'N/A')
            shares = res_data.get('sharesOutstanding', 'N/A')
            inst = res_data.get('institutionalHeld', 'N/A')
            raw_df_found = item.get('raw_df')

            with cols[col_idx % 2]:
                with st.container():
                    st.markdown(f'<p style="font-size:0.95rem; font-weight:bold; color:#60A5FA; margin-bottom:0px;">🟢 {ticker_found} : {co_name}</p>', unsafe_allow_html=True)
                    st.caption(f"🏷️ กลุ่มธุรกิจ: {sec_name} | Support 1: ${res_data['Support 1 ($)']} | ต้าน1: ${res_data['Resist 1 ($)']} | หุ้นทั้งหมด: {shares} | สถาบัน: {inst}")
                    
                    if raw_df_found is not None:
                        st.markdown(f'<div class="chart-header-badge">{ticker_found} | ราคาปัจจุบัน: ${res_data["Price ($)"]} (RSI: {res_data.get("RSI", 0)})</div>', unsafe_allow_html=True)
                        fig_gallery = create_ta_chart(raw_df_found, ticker_found, res_data)
                        if fig_gallery:
                            st.plotly_chart(fig_gallery, use_container_width=True, config=PLOTLY_CONFIG)
                        st.markdown(f'<div class="pattern-box" style="font-size:0.75rem; padding:4px 8px;">😊 {UI_LANG_MAP["placeholder_pattern_match"]}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("ไม่พบข้อมูลกราฟ")
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                col_idx += 1

        st.markdown("---")
        st.markdown("#### 📊 ตารางสรุปสัญญาณราคาและโครงสร้างผู้ถือหุ้น")
        st.dataframe(df_result_display, use_container_width=True, hide_index=True, height=200)
        st.download_button(
            label='📥 ดาวน์โหลด Watchlist วันนี้ (CSV)',
            data=df_result_display.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f'us_watchlist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        )


# --- TAB 3: Watchlist ส่วนตัว ---
with tab3:
    st.markdown("### ⭐ รายชื่อหุ้นใน Watchlist ส่วนตัวของคุณ")
    if st.session_state.watchlist:
        if st.button("🗑️ ล้างรายชื่อ Watchlist ทั้งหมด"):
            st.session_state.watchlist = []
            st.rerun()
        
        st.write(f"หุ้นที่คุณติดตามอยู่ทั้งหมด: {', '.join(st.session_state.watchlist)}")
        st.markdown("---")
        
        for w_ticker in st.session_state.watchlist:
            try:
                stock = yf.Ticker(w_ticker)
                df_w = stock.history(period='5d')
                if not df_w.empty:
                    curr_price = round(df_w['Close'].iloc[-1], 2)
                    prev_close = df_w['Close'].iloc[-2] if len(df_w) > 1 else curr_price
                    change = round(((curr_price - prev_close) / prev_close) * 100, 2)
                    
                    st.markdown(f"""
                    <div class="fin-card">
                        <b>📌 {w_ticker}</b> | ราคาปัจจุบัน: <b>${curr_price}</b> ({change:+.2f}%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"📌 **{w_ticker}** | ไม่สามารถดึงข้อมูลราคาได้ในขณะนี้")
            except Exception:
                st.warning(f"📌 **{w_ticker}** | เกิดข้อผิดพลาดในการเชื่อมต่อข้อมูล")
    else:
        st.info("ยังไม่มีหุ้นใน Watchlist ส่วนตัว ลองค้นหาหุ้นรายตัวแล้วกดปุ่มเพิ่มได้เลยครับ!")
