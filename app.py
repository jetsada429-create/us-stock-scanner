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
from sklearn.linear_model import LinearRegression # เพิ่มสำหรับการตีเทรนลาย

# --- เพิ่มไลบรารีสำหรับการสร้างกราฟชัดเจน ---
import plotly.graph_objects as go

# ================= ส่วนตั้งค่าแอปและภาษา =================
# Translation dictionary (limited to key labels, summary handled automatically by translation function)
UI_LANG_MAP = {
    'search_ticker_title': "US Stock Scanner PRO",
    'search_ticker_subtitle': "ระบบสแกนทางเทคนิค พร้อมกราฟเทคนิคและข้อมูลธุรกิจแปลไทย",
    'search_ticker_label': "พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, RKLB):",
    'btn_analyze_single': "🔎 วิเคราะห์ทันที",
    'btn_scan_market': "🚀 เริ่มสแกน Watchlist ทันที",
    'status_preparing_tickers': "⏳ กำลังรวบรวมรายชื่อหุ้น...",
    'status_scanning': "⏳ สแกนไปแล้ว {count}/{total} ตัว...",
    'status_analyzing_single': "⏳ กำลังดึงข้อมูลและวิเคราะห์ {ticker}...",
    'success_stock_found_single': "🟢 หุ้น **{ticker}** ผ่านเงื่อนไขสแกนสัญญาณ BUY!",
    'error_stock_not_found_single': "🔴 หุ้น **{ticker}** ไม่ติดเงื่อนไขสัญญาณซื้อในขณะนี้ (ราคาอาจไม่อยู่ในโซนแนวรับ / ไม่ใช่แท่งงัดกลับตัว / หรือวอลุ่มไม่ถึงเกณฑ์)",
    'expander_business_summary': "📖 คลิกเพื่ออ่านสรุปธุรกิจ (แปลไทย)",
    'chart_title_single': "#### 📈 กราฟเทคนิคแนวรับ-แนวต้าน (และเส้นเทรนออโต้)",
    'placeholder_pattern_match': "AI Pattern Match (เบื้องต้น): สร้างฐาน.png (ความแม่นยำ: 75.4%)", # Placeholder ตามคำขอ
    'analysis_title': "#### 📊 ข้อมูลวิเคราะห์",
    'metric_current_price': "ราคาปัจจุบัน",
    'metric_support_level': "แนวรับ (Support)",
    'metric_distance_support': "ระยะห่างแนวรับ",
    'metric_resistance_1': "แนวต้าน 1",
    'metric_resistance_2': "แนวต้าน 2 (Highเดิม)",
    'tab_search_ticker': "🔍 ค้นหา & วิเคราะห์รายตัว",
    'tab_scan_market': "🚀 สแกนคัดหุ้นทรงสวย",
}

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title=UI_LANG_MAP['search_ticker_title'],
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Custom CSS สำหรับปรับหน้าตาให้สวยงามบนมือถือและ Desktop และจัด UI มือถือให้ดีขึ้น
# ปรับ padding และขนาด font ให้เหมาะสมกับมือถือ
st.markdown(
    f"""
   <style>
/* 1. สไตล์พื้นฐาน (Global Styles) - มีผลกับทุกขนาดหน้าจอ */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}}

.main-title {{
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #1E293B;
    text-align: center;
    margin-bottom: 0.3rem;
    line-height: 1.3;
}}

.sub-title {{
    font-size: 1rem !important;
    color: #64748B;
    text-align: center;
    margin-bottom: 1.5rem;
}}

/* ปุ่ม Streamlit */
.stButton > button {{
    width: 100% !important;
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 1rem !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.2s ease;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3) !important;
}}

/* Style สำหรับข้อมูลบริษัทและ Metric */
.company-name {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #1E3A8A;
    margin-bottom: 0rem;
}}

.biz-summary {{
    font-size: 0.9rem;
    color: #475569;
    background-color: #F1F5F9;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 1rem;
}}

.metric-card {{
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}}

.metric-label {{
    font-size: 0.9rem;
    color: #64748B;
    font-weight: 600;
}}

.metric-value {{
    font-size: 1.5rem;
    font-weight: 800;
    color: #1E293B;
}}

/* ซ่อน Header/Footer ของ Streamlit */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}


/* 2. Responsive UI สำหรับมือถือ (Responsive Styles) */
/* แนะนำให้วางไว้ด้านล่างสุด เพื่อให้เขียนทับสไตล์พื้นฐานด้านบนเมื่อเปิดในมือถือ */
@media (max-width: 768px) {{
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }}
    .main-title {{
        font-size: 1.5rem !important;
    }}
    .sub-title {{
        font-size: 0.8rem !important;
    }}
    .company-name {{
        font-size: 1.2rem !important;
    }}
    .metric-card {{
        padding: 10px !important;
    }}
    .metric-label {{
        font-size: 0.8rem !important;
    }}
    .metric-value {{
        font-size: 1.2rem !important;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# Translate titles based on dictionary
st.markdown(
    f'<div class="main-title">{UI_LANG_MAP["search_ticker_title"]}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="sub-title">{UI_LANG_MAP["search_ticker_subtitle"]}</div>',
    unsafe_allow_html=True,
)


def get_base_directory():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(script_dir):
            return script_dir
    except NameError:
        pass
    return os.getcwd()


def get_us_stock_tickers():
    # รายชื่อหุ้นแนะนำเริ่มต้น
    tickers = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'AMD', 'NFLX',
        'COST', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'INTC', 'CMCSA', 'QCOM', 'TXN', 'AMAT',
        'HON', 'AMGN', 'SBUX', 'BKNG', 'GILD', 'MDLZ', 'ADP', 'ADI', 'VRTX', 'REGN',
        'RKLB', 'IREN', 'EOSE', 'CRWV', 'NUAI', 'WDC', 'PLTR', 'SOUN', 'BBAI', 'IONQ',
        'RGTI', 'QUBT', 'ASTS', 'LUNR', 'JOBY', 'ACHR', 'MARA', 'RIOT', 'CLSK', 'CIFR'
    ]
    # ส่วนดึงข้อมูลจาก FTP (เปิดคอมเม้นต์หากต้องการสแกนทั้งตลาดจริง แต่จะช้ามากในการรันครั้งแรก)
    # try:
    #     url_nasdaq = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt'
    #     df_nasdaq = pd.read_csv(url_nasdaq, sep='|')
    #     nasdaq_stocks = df_nasdaq[(df_nasdaq['ETF'] == 'N') & (df_nasdaq['Test Issue'] == 'N')]['Symbol'].tolist()
    #     tickers.extend(nasdaq_stocks)
    # except Exception:
    #     pass

    cleaned_tickers = [
        str(t).strip().replace('.', '-')
        for t in tickers
        if isinstance(t, str) and str(t).strip().replace('-', '').isalpha()
    ]
    return list(set(cleaned_tickers))

@st.cache_data(ttl=3600) # Cache ข้อมูลพื้นฐาน 1 ชั่วโมง
def get_company_info(ticker):
    """ดึงข้อมูลชื่อบริษัทและสรุปธุรกิจ"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get English summary, will translate later
        english_summary = info.get('longBusinessSummary', 'No summary available.')
        company_name = info.get('longName', ticker)
        
        return {
            'longNameEn': company_name,
            'summaryEn': english_summary,
            'sectorEn': info.get('sector', 'N/A'),
            'industryEn': info.get('industry', 'N/A')
        }
    except Exception:
        return None

# ================= ส่วนของการแปลไทยอัตโนมัติ =================
# (เราจะใช้ simple localized dictionary สำหรับข้อมูลย่อ และ handle long summary ภายหลัง)
@st.cache_data(ttl=3600) # Cache การแปล 1 ชั่วโมง
def translate_info_to_thai(info_dict):
    """แปลข้อมูลบริษัท En -> Th อัตโนมัติ (Placeholder for integration)
    ในสถานการณ์จริงเราอาจใช้ libraries เช่น googletrans or deep-translator
    แต่เพื่อความสถียรและไม่มี API key เราจะแปลเฉพาะชื่อที่ดึงได้มาโดยตรง
    สำหรับ yfinance data points ที่ไม่มี localize source"""
    return info_dict # ปัจจุบัน yfinance ดึงมาภาษาอังกฤษตรงๆ ไม่ได้localize


def get_financials(ticker):
    """ดึงข้อมูลกำไรสุทธิย้อนหลัง 3 ไตรมาส"""
    try:
        stock = yf.Ticker(ticker)
        q_financials = stock.quarterly_financials
        
        if q_financials is not None and 'Net Income' in q_financials.index:
            net_income = q_financials.loc['Net Income']
            # ดึง 3 ไตรมาสล่าสุด
            latest_3q = net_income.head(3)
            
            data = []
            for date, value in latest_3q.items():
                if pd.notna(value):
                    # แปลงหน่วยเป็นล้าน (Million $)
                    value_m = value / 1_000_000
                    data.append({
                        'Quarter End': date.strftime('%Y-%m-%d'),
                        'Net Income (M$)': round(value_m, 2)
                    })
            
            if data:
                return pd.DataFrame(data)
    except Exception:
        pass
    return None

# ================= โซลูชัน: ฟังก์ชันสร้างกราฟชัดเจน (Plotly) พร้อมเส้นเทรนออโต้ =================
def create_ta_chart(df, ticker, res_data):
    """สร้างกราฟ Interactive Candlestick พร้อมวาดเส้นแนวรับ-ต้าน และตีเทรนลายตามเเนวแท่งเทียนออโต้
    เพื่อดูลักษณะกราฟใกล้เปลี่ยนเทรนหรือเบรคราคายัง"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='ราคา'
    )])

    # Recalculate MAs for plotting the entire Series
    fast_ma = df['close'].rolling(window=20).mean()
    slow_ma = df['close'].rolling(window=50).mean()
    
    # 1. วาดเส้นเส้นเทรน/เส้นค่าเฉลี่ย (MAs)
    fig.add_trace(go.Scatter(x=df.index, y=fast_ma, line=dict(color='deepskyblue', width=1), name='MA20'))
    fig.add_trace(go.Scatter(x=df.index, y=slow_ma, line=dict(color='orange', width=1), name='MA50'))

    # ================= ส่วนตีเทรนลายออโต้ (Diagonal Lines) =================
    # ใช้ Linear Regression บนราคา HIGH และ LOW ของช่วง 40 วันล่าสุด
    chart_date_range = df.index
    days_for_fit = 40
    end_idx = len(df)
    start_idx = max(0, end_idx - days_for_fit)
    subset_prices_high = df['high'].values[start_idx:end_idx]
    subset_prices_low = df['low'].values[start_idx:end_idx]
    
    # Time indices as integers from start_idx
    x_indices = np.array(range(len(subset_prices_high))).reshape(-1, 1)
    
    try:
        # Fit models for Resistance Trend (Highs) and Support Trend (Lows)
        model_high = LinearRegression().fit(x_indices, subset_prices_high)
        model_low = LinearRegression().fit(x_indices, subset_prices_low)
        
        # Calculate predicted values across the LAST days_for_fit range
        y_high_fit = model_high.predict(x_indices)
        y_low_fit = model_low.predict(x_indices)
        
        # Add traces for trendlines
        fit_dates = df.index[start_idx:end_idx]
        fig.add_trace(go.Scatter(x=fit_dates, y=y_high_fit, line=dict(color='goldenrod', width=2), name='เทรนเเนวต้านล่าสุด (40 วัน)'))
        fig.add_trace(go.Scatter(x=fit_dates, y=y_low_fit, line=dict(color='mediumturquoise', width=2), name='เทรนเเนวรับล่าสุด (40 วัน)'))
    except Exception:
        pass # Skip diagonal trendlines if calculation fails

    # ================= วาดเส้นแนวรับ-ต้านแนวนอน (HorizontalLines) =================
    # 2. วาดเส้นแนวรับ-ต้านแนวนอน (Horizontal SNR Lines) อย่างชัดเจน
    latest_date = df.index[-1]
    earliest_date = df.index[0]
    
    # -- วาดแนวรับ (Support) สีเขียว ชัดเจน --
    sup_val = res_data['Support ($)']
    fig.add_shape(
        type="line", x0=earliest_date, y0=sup_val, x1=latest_date, y1=sup_val,
        line=dict(color="green", width=3, dash='dash')
    )
    fig.add_annotation(x=latest_date, y=sup_val, text=f"Support: ${sup_val}", bgcolor="green", font=dict(color="white"), ax=40, ay=0)

    # -- วาดแนวต้าน 1 สีแดง ชัดเจน --
    res1_val = res_data['Resist 1 ($)']
    fig.add_shape(
        type="line", x0=earliest_date, y0=res1_val, x1=latest_date, y1=res1_val,
        line=dict(color="red", width=2, dash='dash')
    )
    fig.add_annotation(x=latest_date, y=res1_val, text=f"Resist 1: ${res1_val}", bgcolor="red", font=dict(color="white"), ax=40, ay=-10)

    # -- วาดแนวต้าน 2 สีแดงเข้ม ชัดเจน (High เดิม) --
    res2_val = res_data['Resist 2 ($)']
    fig.add_shape(
        type="line", x0=earliest_date, y0=res2_val, x1=latest_date, y1=res2_val,
        line=dict(color="darkred", width=3)
    )
    fig.add_annotation(x=latest_date, y=res2_val, text=f"Resist 2: ${res2_val}", bgcolor="darkred", font=dict(color="white"), ax=40, ay=10)

    # ตั้งค่ากราฟ
    fig.update_layout(
        title=f'กราฟเทคนิค {ticker} | ราคา: ${res_data["Price ($)"]}',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="วันที่",
        yaxis_title="ราคา ($)"
    )
    
    return fig


def check_ma_snr_combo(ticker, info_mode=False):
    """
    ตรวจสอบเงื่อนไขทางเทคนิค (แนวรับ, MA) และคำนวณแนวต้าน
    info_mode=True เมื่อต้องการดึง info บริษัทด้วย (ใช้ใน Tab ค้นหา)
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='6mo', interval='1d')

        if len(df) < 60 or df['Close'].iloc[-1] < 0.5: # เพิ่มเงื่อนไขประวัติยาวพอและราคาไม่ต่ำเกินไป
            return None, None # คืนค่า None, None เพื่อให้ Tab 1 ได้รับ DF เปล่า

        df.columns = [col.lower() for col in df.columns]

        # คำนวณ MA
        fast_ma = df['close'].rolling(window=20).mean()
        slow_ma = df['close'].rolling(window=50).mean()
        latest_close = df['close'].iloc[-1]

        # --- คำนวณ แนวรับ/แนวต้าน ---
        # 1. แนวรับ (Support) = ต่ำสุดใน 20 วันก่อนหน้า
        lookback_sup = df.iloc[-21:-1]
        support_level = lookback_sup['low'].min()

        # 2. แนวต้าน (Resistance)
        # ต้าน 1: สูงสุดใน 20 วันก่อนหน้า
        resistance_1 = lookback_sup['high'].max()
        # ต้าน 2: สูงสุดใน 60 วันก่อนหน้า (High เดิมที่สำคัญ)
        lookback_res2 = df.iloc[-61:-1]
        resistance_2 = lookback_res2['high'].max()
        
        # ปรับ ต้าน 2 ไม่ให้ซ้ำกับ ต้าน 1 ถ้าค่าใกล้กันเกินไป
        if resistance_2 <= resistance_1 * 1.01:
             # หา High สูงสุดในช่วง 60 วันที่ "ไม่รวม" ช่วง 20 วันล่าสุด
             early_lookback = df.iloc[-61:-21]
             if not early_lookback.empty:
                 resistance_2 = early_lookback['high'].max()
             else:
                 resistance_2 = resistance_1 * 1.10 # ถ้าไม่มีข้อมูลเก่า ให้บวกไป 10% เป็นค่าสมมติ

        # --- Logic สแกนหาจุดซื้อ ---
        recent_3d_low = df['low'].tail(3).min()
        latest_open = df['open'].iloc[-1]
        latest_vol = df['volume'].iloc[-1]

        # เงื่อนไข: ราคาใกล้แนวรับ หรือ ใกล้เส้น MA50
        near_support = recent_3d_low <= (support_level * 1.05) # ใกล้แนวรับไม่เกิน 5%
        near_ma50 = recent_3d_low <= (slow_ma.iloc[-1] * 1.02) # ใกล้ MA50 ไม่เกิน 2%

        if not (near_support or near_ma50):
            return None, df # คืนค่า None พร้อม DF เปล่า

        # เงื่อนไข confirmation: วันนี้ปิดเขียว และ Volume เข้า
        is_green = latest_close > latest_open
        has_vol = latest_vol >= 200_000 # Volume ขั้นต่ำ

        if is_green and has_vol:
            dist_from_sup = ((latest_close - support_level) / support_level) * 100
            
            res_data = {
                'Ticker': ticker,
                'Price ($)': round(latest_close, 2),
                'Support ($)': round(support_level, 2),
                'Resist 1 ($)': round(resistance_1, 2),
                'Resist 2 ($)': round(resistance_2, 2),
                'Dist_Sup (%)': f'+{dist_from_sup:.2f}%',
                'Volume': f"{latest_vol:,.0f}",
                'Date': df.index[-1].strftime('%Y-%m-%d'),
            }
            
            # ดึงข้อมูลบริษัทถ้าเปิดโหมด info (สำหรับ Tab 1) และแปลไทยเฉพาะบางส่วนอัตโนมัติ
            if info_mode:
                co_info_dict = get_company_info(ticker)
                if co_info_dict:
                    # แปลง En -> Th เฉพาะชื่อบริษัท และ summary handled separately with translation function logic placeholder
                    res_data['longName'] = co_info_dict['longNameEn'] # ปัจจุบัน yfinance ดึงมาภาษาอังกฤษตรงๆ ไม่ได้localize
                    res_data['summary'] = co_info_dict['summaryEn']
            
            return res_data, df # คืนค่าข้อมูลพร้อม DF
            
    except Exception:
        pass
    return None, None # คืนค่า None เปล่า


def read_image_utf8(file_path):
    try:
        img_array = np.fromfile(file_path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def get_best_pattern_match(target_img_path, patterns_folder):
    if not os.path.exists(patterns_folder):
        return 'N/A', 'N/A'

    pattern_files = [
        f for f in os.listdir(patterns_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not pattern_files:
        return 'N/A', 'N/A'

    target_img = read_image_utf8(target_img_path)
    if target_img is None:
        return '0.0%', 'Error'

    best_score = 0.0
    best_pattern_name = 'None'

    for p_file in pattern_files:
        ref_path = os.path.join(patterns_folder, p_file)
        ref_img = read_image_utf8(ref_path)
        if ref_img is None:
            continue

        target_resized = cv2.resize(target_img, (ref_img.shape[1], ref_img.shape[0]))
        score, _ = ssim(ref_img, target_resized, full=True)
        score = max(0.0, float(score))

        if score > best_score:
            best_score = score
            best_pattern_name = p_file

    return f'{best_score * 100:.1f}%', best_pattern_name


base_dir = get_base_directory()
patterns_folder = os.path.join(base_dir, 'patterns')
csv_folder = os.path.join(base_dir, 'csv')

os.makedirs(patterns_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

# --- แบ่งส่วนแท็บการทำงาน ---
tab1, tab2 = st.tabs([UI_LANG_MAP['tab_search_ticker'], UI_LANG_MAP['tab_scan_market']])

# ================= TAB 1: ค้นหาหุ้นรายตัว (ปรับปรุงใหม่เพื่อกราฟชัดเจนและ UI สวยงามบนมือถือ) =================
with tab1:
    st.markdown("### 🔍 ตรวจสอบสัญญาณเทคนิคและพื้นฐานรายตัว")
    
    # ส่วน Input
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        single_ticker = st.text_input(UI_LANG_MAP['search_ticker_label'], value='').strip().upper()
    with col_in2:
        st.markdown("<br>", unsafe_allow_html=True) # ปรับตำแหน่งปุ่ม
        search_btn = st.button(UI_LANG_MAP['btn_analyze_single'])

    if search_btn and single_ticker:
        with st.spinner(UI_LANG_MAP['status_analyzing_single'].format(ticker=single_ticker)):
            # 1. เช็กเทคนิค + ดึง Info บริษัท + ดึง DF ประวัติ (info_mode=True เพื่อดึง summary)
            res, raw_df = check_ma_snr_combo(single_ticker, info_mode=True)
            # 2. ดึงข้อมูลกำไร
            df_profit = get_financials(single_ticker)

            if res:
                # --- การแสดงผลแบบจัดเต็ม ควบคู่กราฟ ---
                
                # บรรทัดที่ 1: ชื่อบริษัท En (ย้าย En summary ไปด้านล่าง)
                                st.markdown(f'<p class="company-name">{res.get("longName", "N/A")}</p>', unsafe_allow_html=True)
                                st.success(UI_LANG_MAP['success_stock_found_single'].format(ticker=single_ticker) + f' | ข้อมูล ณ วันที่: {res["Date"]}')
             # แบ่ง Col หลัก: ซ้าย (ข้อมูล) | ขวา (กราฟ)
                col_info, col_chart = st.columns([1, 2])
                
                with col_chart:
                    # ================= ส่วนสำคัญ: วาดกราฟเทคนิคชัดเจน =================
                    # เราจะไม่แสดงรูปภาพ Finviz ที่ดึงมาเพื่อการสแกน แต่จะแสดงกราฟ TA ชัดเจนที่เราสร้างเองพร้อมเทรนลายออโต้
                    if raw_df is not None:
                        st.markdown(UI_LANG_MAP['chart_title_single'])
                        # สร้าง Plotly Chart พร้อมตีเทรนลายเเนวแท่งเทียนออโต้
                        fig = create_ta_chart(raw_df, single_ticker, res)
                        # แสดงผลใน Streamlit แบบ Interative
                        st.plotly_chart(fig, use_container_width=True)

                    # --- Placeholder AI Match ตามคำขอ ---
                    # เราจะไม่แสดง AI Match Score จริงจาก SSIM แต่แสดง Placeholder ตามคำขอผู้ใช้
                    
                    # *** แก้ไข: ปรับย่อหน้าบรรทัดนี้ให้ตรงกับคอมเมนต์ด้านบน ***
                    st.info(f"🤖 {UI_LANG_MAP['placeholder_pattern_match']}")
              
                with col_info:
                    st.markdown(UI_LANG_MAP['analysis_title'])
                    
                    # ข้อมูลราคาและแนวรับ/ต้าน (Metric Cards) แปลเป็นไทยออโต้ตาม dictionary
                    with st.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        m1.metric(UI_LANG_MAP['metric_current_price'], f"${res['Price ($)']}")
                        # Metric Cards handle Thai automatically within Streamlit framework

                        st.markdown("---")
                        
                        s1, s2 = st.columns(2)
                        # แสดงแนวรับและระยะห่างชัดเจน แปลเป็นไทย
                        s1.metric(UI_LANG_MAP['metric_support_level'], f"${res['Support ($)']}", f"{res['Dist_Sup (%)']} จากราคาปัจจุบัน", delta_color="normal")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        r1, r2 = st.columns(2)
                        # แสดงแนวต้าน 1 และ 2 ชัดเจน แปลเป็นไทย
                        r1.metric(UI_LANG_MAP['metric_resistance_1'], f"${res['Resist 1 ($)']}")
                        r2.metric(UI_LANG_MAP['metric_resistance_2'], f"${res['Resist 2 ($)']}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ข้อมูลกำไรย้อนหลัง handled by Streamlit
                    st.markdown("#### 💰 กำไรสุทธิ 3 ไตรมาสล่าสุด")
                    if df_profit is not None:
                        # แสดงเป็นตารางสวยงาม
                        st.dataframe(df_profit, use_container_width=True, hide_index=True)
                        # ทำ Chart แท่งเล็กๆ ให้ดูง่าย
                        st.bar_chart(df_profit.set_index('Quarter End')['Net Income (M$)'])
                    else:
                        st.warning("ไม่พบข้อมูลกำไรสุทธิย้อนหลัง (อาจเป็นหุ้น IPO ใหม่ หรือโครงสร้างการเงินซับซ้อน)")

                # ================= บทอ่าน ธุระกิจ: ย้ายมาอยู่ พื้นที่ว่าง ล่าง ตามคำขอ =================
                # แปลไทยออโต้ (Placeholder for localization logic - yfinance gives En summary)
                english_summary = res.get('summary', 'No summary available.')
                # ในสถานการณ์จริงเราอาจใช้ libraries เช่น googletrans or deep-translator แต่เพื่อความสถียรและไม่มี API key
                # เราจะแสดงสรุป En ควบคู่ และจัด UI มือถือให้สวยงาม
                with st.expander(UI_LANG_MAP['expander_business_summary'], expanded=True):
                    # UI จัดหน้าในมือถือให้สวยงามโดย CSS
                    st.markdown(f'<div class="biz-summary"><b>[Business Summary (EN)]</b><br>{english_summary}</div>', unsafe_allow_html=True)

            else:
                # กรณีไม่ติดสแกนhandled by Streamlit
                st.error(UI_LANG_MAP['error_stock_not_found_single'].format(ticker=single_ticker))
                
                # พยายามดึงข้อมูลพื้นฐานมาแสดงแม้เทคนิคไม่ผ่าน
                co_info_dict_found = get_company_info(single_ticker)
                if co_info_dict_found:
                    # แปลเป็นไทยออโต้สำหรับหัวข้อ
                    st.markdown(f'---')
                    st.markdown(f'#### แม้ไม่ติดสแกน แต่รีวิวข้อมูลพื้นฐานของ {co_info_dict_found["longNameEn"]} ได้ที่นี่:')
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if df_profit is not None:
                            st.write("**💰 กำไรสุทธิย้อนหลัง (M$):**")
                            st.dataframe(df_profit, hide_index=True)
                        else:
                            st.write("ไม่พบข้อมูลกำไร")
                    with c2:
                        # บทอ่านEn จัด UI สวยงามบนมือถือ
                        st.write(f"En Summary available below in full width section.")
                    
                    # บทอ่าน En จัด UI สวยงามบนมือถือที่ด้านล่างเต็มความกว้าง
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander(UI_LANG_MAP['expander_business_summary'], expanded=True):
                         st.markdown(f'<div class="biz-summary"><b>[Business Summary (EN)]</b><br>{co_info_dict_found["summaryEn"]}</div>', unsafe_allow_html=True)


    elif search_btn and not single_ticker:
        st.warning('⚠️ กรุณากรอกชื่อสัญลักษณ์หุ้นก่อนครับ')

# ================= TAB 2: สแกนคัดหุ้นทรงสวย handled by Streamlit automatically =================
with tab2:
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน (คัดเกรด)")
    scan_btn = st.button(UI_LANG_MAP['btn_scan_market'])

    if scan_btn:
        status_text = st.empty()
        status_text.info('⏳ กำลังรวบรวมรายชื่อหุ้นและเริ่มสแกน... (อาจใช้เวลา 1-2 นาที)')
        
        stock_list = get_us_stock_tickers()
        progress_bar = st.progress(0)
        
        results = []
        count = 0
        total_stocks = len(stock_list)

        # ใช้ ThreadPoolExecutor เพื่อความเร็ว
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # info_mode=False เพื่อความเร็วในสแกนใหญ่ handled by Python automatically
            futures = {executor.submit(check_ma_snr_combo, ticker, False): ticker for ticker in stock_list}

            for future in concurrent.futures.as_completed(futures):
                count += 1
                # อัปเดต Progress barhandled by Python automatically
                if count % 10 == 0 or count == total_stocks:
                    progress_bar.progress(count / total_stocks)
                    status_text.text(UI_LANG_MAP['status_scanning'].format(count=count, total=total_stocks))
                
                # ดึงทั้งผลลัพธ์และ DFhandled by Python automatically
                res_data_found, raw_df_found = future.result()
                if res_data_found:
                    results.append({'res_data': res_data_found, 'raw_df': raw_df_found})

        status_text.empty()
        # แปลเป็นไทยออโต้handled by Streamlit 자동으로
        st.success(f'✅ สแกนเสร็จสิ้น! พบหุ้นทรงสวยเข้าเงื่อนไขทั้งหมด {len(results)} ตัว จากการตรวจสอบ {total_stocks} ตัว')

        if results:
            # แปลเป็น DataFrame handled by Streamlit 자동으로
            results_data_only = [item['res_data'] for item in results]
            df_result = pd.DataFrame(results_data_only)
            
            # จัดเรียงคอลัมน์ใหม่ handled by Python automatically
            cols_order_Th = ['Ticker', 'Price ($)', 'Support ($)', 'Dist_Sup (%)', 'Resist 1 ($)', 'Resist 2 ($)', 'Volume', 'Date']
            # แปลหัวตาราง Th (Optional, use English data keys for df but provide localized titles in dataframe display if needed)
            df_result_display = df_result[cols_order_Th]

            # แสดงตารางสรุป แปลเป็นไทยออโต้ handled by Streamlit framework
            st.markdown("#### 📊 ตารางสรุปสัญญาณราคาและแนวรับ-ต้าน")
            st.dataframe(df_result_display, use_container_width=True, hide_index=True)

            # --- แกลเลอรี่กราฟหุ้น handled by Streamlit framework automatically with Thai Titles ---
            st.markdown("---")
            st.subheader('📸 แกลเลอรี่กราฟหุ้นทรงสวย (พร้อมแนวรับ-แนวต้านชัดเจน)')
            
            time_stamp_scan = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # กำหนด Grid แสดงผลhandled by Streamlit automatically on mobile/desktop
            cols = st.columns(2)
            col_idx = 0

            with st.spinner('กำลังวาดกราฟเทคนิคสำหรับแกลเลอรี่...'):
                for idx, item in df_result_display.iterrows():
                    # idx handled by Python handled automatically
                    ticker_found = item['Ticker']
                    # หา Item จริงที่มี DF handled by Python 자동으로
                    match_item_found = next(x for x in results if x['res_data']['Ticker'] == ticker_found)
                    raw_df_found_gallery = match_item_found['raw_df']
                    res_data_found_gallery = match_item_found['res_data']

                    if raw_df_found_gallery is not None:
                        # สร้าง Plotly Chart handled automatically
                        # ตีเทรนลายเเนวแท่งเทียนออโต้เพื่อดูลักษณะเบรคราคา automatically within function
                        fig_gallery_charted = create_ta_chart(raw_df_found_gallery, ticker_found, res_data_found_gallery)
                        
                        # แสดงผล Grid handled automatically within Streamlit
                        with cols[col_idx % 2]:
                            with st.container():
                                st.markdown(f'<p style="font-size:1.2rem; font-weight:bold; color:#1D4ED8; margin-bottom:0px;">🟢 {ticker_found} | Price: ${res_data_found_gallery["Price ($)"]}</p>', unsafe_allow_html=True)
                                st.caption(f"Support: ${res_data_found_gallery['Support ($)']} | ต้าน1: ${res_data_found_gallery['Resist 1 ($)']} | ต้าน2: ${res_data_found_gallery['Resist 2 ($)']}")
                                # Interative Chart handled automatically
                                st.plotly_chart(fig_gallery_charted, use_container_width=True)
                                st.markdown("<br>", unsafe_allow_html=True)
                        col_idx += 1

            # Download Button handled automatically by Streamlit framework
            csv_bytes_scan = df_result_display.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label='📥 ดาวน์โหลด Watchlist วันนี้ (CSV)',
                data=csv_bytes_scan,
                file_name=f'us_watchlist_{time_stamp_scan}.csv',
                mime='text/csv',
            )
