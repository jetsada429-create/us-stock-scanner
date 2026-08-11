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
# --- เพิ่มไลบรารีสำหรับการสร้างกราฟชัดเจน ---
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="US Stock Scanner PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Custom CSS สำหรับปรับหน้าตาให้สวยงามบนมือถือและ Desktop
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .main-title {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.3rem;
        line-height: 1.3;
    }
    .sub-title {
        font-size: 1rem !important;
        color: #64748B;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
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
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3) !important;
    }
    /* Style สำหรับข้อมูลบริษัท */
    .company-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0rem;
    }
    .biz-summary {
        font-size: 0.9rem;
        color: #475569;
        background-color: #F1F5F9;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">📈 สแกนหุ้นสหรัฐฯ PRO</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">ระบบสแกนทางเทคนิค พร้อมกราฟเทคนิคแนวรับ-แนวต้านชัดเจน</div>',
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
        info = yf.Ticker(ticker).info
        return {
            'longName': info.get('longName', 'N/A'),
            'longBusinessSummary': info.get('longBusinessSummary', 'No summary available.'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A')
        }
    except Exception:
        return None

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

# ================= โซลูชัน: ฟังก์ชันสร้างกราฟชัดเจน (Plotly) =================
def create_ta_chart(df, ticker, res_data):
    """สร้างกราฟ Interactive Candlestick พร้อมวาดเส้นแนวรับ-ต้านอย่างชัดเจน"""
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

    # -- วาดแนวต้าน 2 สีแดงเข้ม ชัดเจน (แนวชน) --
    res2_val = res_data['Resist 2 ($)']
    fig.add_shape(
        type="line", x0=earliest_date, y0=res2_val, x1=latest_date, y1=res2_val,
        line=dict(color="darkred", width=3)
    )
    fig.add_annotation(x=latest_date, y=res2_val, text=f"Resist 2: ${res2_val}", bgcolor="darkred", font=dict(color="white"), ax=40, ay=10)

    # ตั้งค่ากราฟ
    fig.update_layout(
        title=f'กราฟเทคนิคคัดสัญญาณ {ticker} | ราคา: ${res_data["Price ($)"]}',
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
            
            # ดึงข้อมูลบริษัทถ้าเปิดโหมด info (สำหรับ Tab 1)
            if info_mode:
                co_info = get_company_info(ticker)
                if co_info:
                    res_data.update(co_info)
            
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
tab1, tab2 = st.tabs(['🔍 ค้นหา & วิเคราะห์รายตัว', '🚀 สแกนหุ้นคัดเกรด (Watchlist)'])

# ================= TAB 1: ค้นหาหุ้นรายตัว (ปรับปรุงใหม่เพื่อกราฟชัดเจน) =================
with tab1:
    st.markdown("### 🔍 ตรวจสอบสัญญาณเทคนิคและพื้นฐานรายตัว")
    
    # ส่วน Input
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        single_ticker = st.text_input('พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, RKLB):', value='').strip().upper()
    with col_in2:
        st.markdown("<br>", unsafe_allow_html=True) # ปรับตำแหน่งปุ่ม
        search_btn = st.button('🔎 วิเคราะห์ทันที')

    if search_btn and single_ticker:
        with st.spinner(f'กำลังดึงข้อมูลและวิเคราะห์ {single_ticker}...'):
            # 1. เช็กเทคนิค + ดึง Info บริษัท + ดึง DF ประวัติ
            res, raw_df = check_ma_snr_combo(single_ticker, info_mode=True)
            # 2. ดึงข้อมูลกำไร
            df_profit = get_financials(single_ticker)

            if res:
                # --- การแสดงผลแบบจัดเต็ม ควบคู่กราฟ ---
                
                # บรรทัดที่ 1: ชื่อบริษัทและ Sector
                st.markdown(f'<p class="company-name">{res["longName"]} ({single_ticker})</p>', unsafe_allow_html=True)
                st.caption(f"Sector: {res['sector']} | Industry: {res['industry']}")
                st.success(f'🟢 สัญญาณทางเทคนิค: **ผ่านเกณฑ์ (BUY Signal)** | ข้อมูล ณ วันที่: {res["Date"]}')

                # แบ่ง Col หลัก: ซ้าย (ข้อมูล) | ขวา (กราฟ)
                col_info, col_chart = st.columns([1, 2])
                
                with col_chart:
                    # ================= ส่วนสำคัญ: วาดกราฟเทคนิคชัดเจน =================
                    # เราจะไม่แสดงรูปภาพ Finviz ที่ดึงมาเพื่อการสแกน แต่จะแสดงกราฟ TA ชัดเจนที่เราสร้างเอง
                    if raw_df is not None:
                        st.markdown("#### 📈 กราฟเทคนิคชัดเจน (แนวรับ-ต้าน)")
                        # สร้าง Plotly Chart
                        fig = create_ta_chart(raw_df, single_ticker, res)
                        # แสดงผลใน Streamlit แบบ Interative
                        st.plotly_chart(fig, use_container_width=True)

                    # --- ส่วนแสดง AI Match (ข้อมูลประกอบ) ---
                    # เรายังโหลด Finviz Image เพื่อให้ AI คำนวณ Match Score ได้เหมือนเดิม
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    chart_url = f'https://charts2.finviz.com/chart.ashx?t={single_ticker}&ty=c&ta=1&p=d&s=l'
                    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    temp_chart_path = os.path.join(base_dir, f'{single_ticker}_single_{time_stamp}.png')

                    try:
                        r = requests.get(chart_url, headers=headers, timeout=10)
                        if r.status_code == 200:
                            with open(temp_chart_path, 'wb') as f:
                                f.write(r.content)
                            
                            # คำนวณ AI Match Score
                            score_str, pattern_name = get_best_pattern_match(temp_chart_path, patterns_folder)
                            st.info(f"🤖 AI Pattern Match (เบื้องต้น): **{pattern_name}** (ความแม่นยำ: {score_str})")
                            
                            # ลบไฟล์ชั่วคราว
                            os.remove(temp_chart_path)
                    except Exception:
                        pass # ข้าม AI ถ้าโหลดรูปไม่ได้

                with col_info:
                    st.markdown("#### 📊 ข้อมูลวิเคราะห์")
                    
                    # ข้อมูลราคาและแนวรับ/ต้าน (Metric Cards)
                    with st.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        m1.metric("ราคาปัจจุบัน", f"${res['Price ($)']}")
                        m2.metric("Volume", res['Volume'])
                        
                        st.markdown("---")
                        
                        s1, s2 = st.columns(2)
                        # แสดงแนวรับและระยะห่างชัดเจน
                        s1.metric("แนวรับ (Support)", f"${res['Support ($)']}", f"{res['Dist_Sup (%)']} จากราคาปัจจุบัน", delta_color="normal")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        r1, r2 = st.columns(2)
                        # แสดงแนวต้าน 1 และ 2 ชัดเจน
                        r1.metric("แนวต้าน 1", f"${res['Resist 1 ($)']}")
                        r2.metric("แนวต้าน 2 (แนวชน)", f"${res['Resist 2 ($)']}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ข้อมูลกำไรย้อนหลัง
                    st.markdown("#### 💰 กำไรสุทธิ 3 ไตรมาสล่าสุด")
                    if df_profit is not None:
                        # แสดงเป็นตารางสวยงาม
                        st.dataframe(df_profit, use_container_width=True, hide_index=True)
                        # ทำ Chart แท่งเล็กๆ ให้ดูง่าย
                        st.bar_chart(df_profit.set_index('Quarter End')['Net Income (M$)'])
                    else:
                        st.warning("ไม่พบข้อมูลกำไรสุทธิย้อนหลัง (อาจเป็นหุ้น IPO ใหม่ หรือโครงสร้างการเงินซับซ้อน)")

                # ส่วนสรุปธุรกิจ (Expander เพื่อไม่ให้รก)
                with st.expander("📖 คลิกเพื่ออ่านสรุปธุรกิจ"):
                    st.markdown(f'<div class="biz-summary">{res["longBusinessSummary"]}</div>', unsafe_allow_html=True)

            else:
                # กรณีไม่ติดสแกน แต่ยังอยากเห็นข้อมูลพื้นฐานและกำไร
                st.error(f'🔴 **{single_ticker}** ไม่ติดเงื่อนไขสัญญาณซื้อทางเทคนิคในขณะนี้ (ราคาอาจไม่อยู่ในโซนแนวรับ หรือไม่ใช่แท่งเทียนงัดกลับตัว)')
                
                # พยายามดึงข้อมูลพื้นฐานมาแสดงแม้เทคนิคไม่ผ่าน
                co_info = get_company_info(single_ticker)
                if co_info:
                    st.markdown(f'---')
                    st.markdown(f'#### แม้ไม่ติดสแกน แต่รีวิวข้อมูลพื้นฐานของ {co_info["longName"]} ได้ที่นี่:')
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if df_profit is not None:
                            st.write("**💰 กำไรสุทธิย้อนหลัง (M$):**")
                            st.dataframe(df_profit, hide_index=True)
                        else:
                            st.write("ไม่พบข้อมูลกำไร")
                    with c2:
                        st.caption(f"Sector: {co_info['sector']} | Industry: {co_info['industry']}")
                        st.markdown(f'<div class="biz-summary" style="font-size:0.8rem;">{co_info["longBusinessSummary"][:500]}...</div>', unsafe_allow_html=True)


    elif search_btn and not single_ticker:
        st.warning('⚠️ กรุณากรอกชื่อสัญลักษณ์หุ้นก่อนครับ')

# ================= TAB 2: สแกนตลาด (Watchlist) =================
with tab2:
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน (คัดเกรด)")
    scan_btn = st.button('🚀 เริ่มสแกน Watchlist ทันที')

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
            # info_mode=False เพื่อความเร็วในสแกนใหญ่
            # เปลี่ยน check_ma_snr_combo ให้คืน DF ด้วย
            futures = {executor.submit(check_ma_snr_combo, ticker, False): ticker for ticker in stock_list}

            for future in concurrent.futures.as_completed(futures):
                count += 1
                # อัปเดต Progress bar ทุกๆ 10 ตัวเพื่อลดโหลด UI
                if count % 10 == 0 or count == total_stocks:
                    progress_bar.progress(count / total_stocks)
                
                # ดึงทั้งผลลัพธ์และ DF
                res, raw_df = future.result()
                if res:
                    # เก็บทั้งผลลัพธ์และ DF ไปใช้ใน Gallery
                    results.append({'res_data': res, 'raw_df': raw_df})

        status_text.empty()
        st.success(f'✅ สแกนเสร็จสิ้น! พบหุ้นทรงสวยเข้าเงื่อนไขทั้งหมด {len(results)} ตัว จากการตรวจสอบ {total_stocks} ตัว')

        if results:
            # แปลงเฉพาะ res_data เป็น DataFrame สำหรับตาราง
            results_data_only = [item['res_data'] for item in results]
            df_result = pd.DataFrame(results_data_only)
            
            # จัดเรียงคอลัมน์ใหม่ให้ดูง่าย
            cols_order = ['Ticker', 'Price ($)', 'Support ($)', 'Dist_Sup (%)', 'Resist 1 ($)', 'Resist 2 ($)', 'Volume', 'Date']
            df_result = df_result[cols_order]

            # แสดงตารางสรุป
            st.markdown("#### 📊 ตารางสรุปสัญญาณราคาและแนวรับ-ต้าน")
            st.dataframe(df_result, use_container_width=True, hide_index=True)

            # --- ส่วนแสดงกราฟของหุ้นที่ติดสแกนใน Gallery ---
            st.markdown("---")
            st.subheader('📸 แกลเลอรี่กราฟหุ้นทรงสวย (พร้อมแนวรับ-แนวต้านชัดเจน)')
            
            time_stamp_scan = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            # กำหนด Grid แสดงผล: 2 คอลัมน์
            cols = st.columns(2)
            col_idx = 0

            with st.spinner('กำลังวาดกราฟเทคนิคสำหรับแกลเลอรี่...'):
                for idx, item in df_result.iterrows():
                    # idx คือ Index ของ DataFrame, แต่เราต้องการข้อมูลจริงจากลิสต์ Results
                    ticker = item['Ticker']
                    # หา Item จริงที่มี DF
                    match_item = next(x for x in results if x['res_data']['Ticker'] == ticker)
                    raw_df = match_item['raw_df']
                    res_data = match_item['res_data']

                    if raw_df is not None:
                        # สร้าง Plotly Chart สำหรับแกลเลอรี่
                        fig_gallery = create_ta_chart(raw_df, ticker, res_data)
                        
                        # แสดงผลใน Grid
                        with cols[col_idx % 2]:
                            with st.container():
                                st.markdown(f'<p style="font-size:1.2rem; font-weight:bold; color:#1D4ED8; margin-bottom:0px;">🟢 {ticker} | Price: ${res_data["Price ($)"]}</p>', unsafe_allow_html=True)
                                st.caption(f"Support: ${res_data['Support ($)']} | ต้าน1: ${res_data['Resist 1 ($)']} | ต้าน2: ${res_data['Resist 2 ($)']}")
                                # แสดง Plotly Chart ในแกลเลอรี่
                                st.plotly_chart(fig_gallery, use_container_width=True)
                                st.markdown("<br>", unsafe_allow_html=True)
                        col_idx += 1

            # ปุ่มดาวน์โหลด CSV
            csv_bytes = df_result.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label='📥 ดาวน์โหลด Watchlist วันนี้ (CSV)',
                data=csv_bytes,
                file_name=f'us_watchlist_{time_stamp_scan}.csv',
                mime='text/csv',
            )
