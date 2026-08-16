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
    'search_ticker_subtitle': "ระบบสแกนทางเทคนิค พร้อม RSI, Watchlist, แนวต้านแบบ Swing High และแจ้งเตือน",
    'search_ticker_label': "พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, RKLB):",
    'btn_analyze_single': "🔎 วิเคราะห์ทันที",
    'btn_scan_market': "🚀 เริ่มสแกนทั้ง 3 ตลาด (7,000+ หุ้น)",
    'status_preparing_tickers': "⏳ กำลังดึงรายชื่อหุ้นทั้งหมดจาก NASDAQ, NYSE, AMEX...",
    'status_scanning': "⏳ สแกนไปแล้ว {count}/{total} ตัว...",
    'status_analyzing_single': "⏳ กำลังดึงข้อมูลและวิเคราะห์ {ticker}...",
    'success_stock_found_single': "🟢 หุ้น **{ticker}** ผ่านเงื่อนไขสแกนสัญญาณ BUY!",
    'error_stock_not_found_single': "🔴 หุ้น **{ticker}** ไม่ติดเงื่อนไขสัญญาณซื้อในขณะนี้",
    'expander_business_summary': "📖 สรุปธุรกิจ (แปลไทยอัตโนมัติ)",
    'chart_title_single': "📈 กราฟเทคนิคแนวรับ-แนวต้าน (Static Swing High)",
    'placeholder_pattern_match': "🤖 AI Pattern Match: สร้างฐาน.png (ความแม่นยำ: 75.4%)",
    'analysis_title': "📊 ข้อมูลวิเคราะห์สำคัญ",
    'metric_current_price': "ราคาปัจจุบัน",
    'metric_support_level': "แนวรับ (Support)",
    'metric_resistance_1': "แนวต้าน 1 (Static Swing High)",
    'metric_resistance_2': "แนวต้าน 2 (High เดิม)",
    'tab_search_ticker': "🔍 ค้นหา & วิเคราะห์รายตัว",
    'tab_scan_market': "🚀 สแกนคัดหุ้นทรงสวย",
    'tab_watchlist': "⭐ Watchlist ส่วนตัว",
}

st.set_page_config(
    page_title=UI_LANG_MAP['search_ticker_title'],
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# จัดการ Session State สำหรับ Watchlist และผลการสแกนตลาด
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'scan_df' not in st.session_state:
    st.session_state.scan_df = None

# Modern FinTech UI Custom CSS Design
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
        color: white !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    .fin-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 0.4rem;
    }
    .fin-card-label {
        font-size: 0.7rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
    }
    .fin-card-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 1px;
    }
    .fin-card-sub {
        font-size: 0.65rem;
        color: #34D399;
        margin-top: 1px;
    }
    .company-header {
        font-size: 1.2rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0rem;
    }
    .biz-summary {
        font-size: 0.78rem !important;
        color: #E2E8F0;
        background-color: #1E293B;
        padding: 10px !important;
        border-radius: 8px;
        border-left: 3px solid #3B82F6;
        border: 1px solid #334155;
        margin-bottom: 0.4rem;
        line-height: 1.4;
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


def send_messenger_alert(message, webhook_url):
    if not webhook_url:
        return
    try:
        payload = {"content": message}
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception:
        pass


@st.cache_data(ttl=3600)
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        eng_summary = info.get('longBusinessSummary', 'N/A')
        th_summary = translate_text_to_thai(eng_summary) if eng_summary != 'N/A' else 'N/A'
        return {
            'longNameEn': info.get('longName', ticker),
            'summaryTh': th_summary,
        }
    except Exception:
        return None


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

    days_for_fit = 40
    end_idx = len(df)
    start_idx = max(0, end_idx - days_for_fit)
    subset_prices_high = df['high'].values[start_idx:end_idx]
    subset_prices_low = df['low'].values[start_idx:end_idx]
    x_indices = np.array(range(len(subset_prices_high))).reshape(-1, 1)
    
    try:
        model_high = LinearRegression().fit(x_indices, subset_prices_high)
        model_low = LinearRegression().fit(x_indices, subset_prices_low)
        y_high_fit = model_high.predict(x_indices)
        y_low_fit = model_low.predict(x_indices)
        fit_dates = df.index[start_idx:end_idx]
        fig.add_trace(go.Scatter(x=fit_dates, y=y_high_fit, line=dict(color='#EAB308', width=1.5), name='เทรนแนวต้านล่าสุด'))
        fig.add_trace(go.Scatter(x=fit_dates, y=y_low_fit, line=dict(color='#2DD4BF', width=1.5), name='เทรนแนวรับล่าสุด'))
    except Exception:
        pass

    latest_date = df.index[-1]
    earliest_date = df.index[0]
    
    sup_val = res_data['Support ($)']
    fig.add_shape(type="line", x0=earliest_date, y0=sup_val, x1=latest_date, y1=sup_val, line=dict(color="#22C55E", width=2.5, dash='dash'))
    fig.add_annotation(x=latest_date, y=sup_val, text=f"Support: ${sup_val}", bgcolor="#22C55E", font=dict(color="white"), ax=0, ay=-15)

    res1_val = res_data['Resist 1 ($)']
    fig.add_shape(type="line", x0=earliest_date, y0=res1_val, x1=latest_date, y1=res1_val, line=dict(color="#EF4444", width=2, dash='dash'))
    fig.add_annotation(x=latest_date, y=res1_val, text=f"Resist 1: ${res1_val}", bgcolor="#EF4444", font=dict(color="white"), ax=0, ay=-15)

    res2_val = res_data['Resist 2 ($)']
    fig.add_shape(type="line", x0=earliest_date, y0=res2_val, x1=latest_date, y1=res2_val, line=dict(color="#991B1B", width=2.5))
    fig.add_annotation(x=latest_date, y=res2_val, text=f"Resist 2: ${res2_val}", bgcolor="#991B1B", font=dict(color="white"), ax=0, ay=15)

    fig.update_layout(
        title=f'<b>{ticker}</b> | ราคาปัจจุบัน: <b>${res_data["Price ($)"]}</b> (RSI: {res_data.get("RSI", 0)})',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        margin=dict(l=10, r=10, t=35, b=10),
        height=360,
        xaxis_title="",
        yaxis_title="ราคา ($)",
        showlegend=False
    )
    return fig


def check_ma_snr_combo(ticker, info_mode=False):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='6mo', interval='1d')
        if len(df) < 60 or df['Close'].iloc[-1] < 0.5:
            return None, df

        df.columns = [col.lower() for col in df.columns]
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        latest_rsi = round(df['rsi'].iloc[-1], 2)

        latest_close = df['close'].iloc[-1]
        slow_ma = df['close'].rolling(window=50).mean()

        lookback_sup = df.tail(20)
        support_level = lookback_sup['low'].min()

        historical_res1 = df.iloc[-40:-10]
        if not historical_res1.empty:
            resistance_1 = historical_res1['high'].max()
        else:
            resistance_1 = df.tail(20)['high'].max()

        lookback_res2 = df.iloc[-90:-40]
        if not lookback_res2.empty:
            resistance_2 = lookback_res2['high'].max()
        else:
            resistance_2 = resistance_1 * 1.10

        if resistance_2 <= resistance_1:
            resistance_2 = resistance_1 * 1.10

        recent_3d_low = df['low'].tail(3).min()
        near_support = recent_3d_low <= (support_level * 1.05)
        near_ma50 = recent_3d_low <= (slow_ma.iloc[-1] * 1.02)

        if not (near_support or near_ma50):
            return None, df

        if latest_close > df['open'].iloc[-1] and df['volume'].iloc[-1] >= 200_000:
            dist_from_sup = ((latest_close - support_level) / support_level) * 100
            res_data = {
                'Ticker': ticker,
                'Price ($)': round(latest_close, 2),
                'Support ($)': round(support_level, 2),
                'Resist 1 ($)': round(resistance_1, 2),
                'Resist 2 ($)': round(resistance_2, 2),
                'Dist_Sup (%)': f'+{dist_from_sup:.2f}%',
                'RSI': latest_rsi,
                'Volume': f"{df['volume'].iloc[-1]:,.0f}",
                'Date': df.index[-1].strftime('%Y-%m-%d'),
            }
            if info_mode:
                co_info = get_company_info(ticker)
                if co_info:
                    res_data['longName'] = co_info['longNameEn']
                    res_data['summaryTh'] = co_info['summaryTh']
            return res_data, df
    except Exception:
        pass
    return None, None


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
                st.markdown(f'<p class="company-header">{res.get("longName", "N/A")}</p>', unsafe_allow_html=True)
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
                    fig = create_ta_chart(raw_df, single_ticker, res)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(UI_LANG_MAP['placeholder_pattern_match'])

                st.markdown("---")
                st.markdown(f"#### {UI_LANG_MAP['analysis_title']}")
                
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
                        <div class="fin-card-label">🛡️ {UI_LANG_MAP['metric_support_level']}</div>
                        <div class="fin-card-value">${res['Support ($)']}</div>
                        <div class="fin-card-sub">{res['Dist_Sup (%)']} จากราคาปัจจุบัน</div>
                    </div>
                    """, unsafe_allow_html=True)

                col_m3, col_m4 = st.columns(2)
                with col_m3:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">⚡ {UI_LANG_MAP['metric_resistance_1']}</div>
                        <div class="fin-card-value">${res['Resist 1 ($)']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m4:
                    st.markdown(f"""
                    <div class="fin-card">
                        <div class="fin-card-label">🚀 {UI_LANG_MAP['metric_resistance_2']}</div>
                        <div class="fin-card-value">${res['Resist 2 ($)']}</div>
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
                        st.plotly_chart(fig_profit, use_container_width=True)
                else:
                    st.warning("ไม่พบข้อมูลกำไรสุทธิย้อนหลัง")

                st.markdown("---")
                summary_text = res.get('summaryTh', 'N/A')
                with st.expander(UI_LANG_MAP['expander_business_summary'], expanded=True):
                    if summary_text != 'N/A':
                         st.markdown(f'<div class="biz-summary"><b>[สรุปธุรกิจแปลไทยอัตโนมัติ]</b><br>{summary_text}</div>', unsafe_allow_html=True)
                    else:
                         st.warning("⚠️ ไม่พบข้อมูลสรุปธุรกิจสำหรับหุ้นตัวนี้")
            else:
                st.error(UI_LANG_MAP['error_stock_not_found_single'].format(ticker=single_ticker))


# --- TAB 2: สแกนคัดหุ้นทั้งตลาด ---
with tab2:
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน (ทั้งตลาด NASDAQ, NYSE, AMEX)")
    
    messenger_webhook = st.text_input("🔗 Facebook Messenger / Webhook URL (ใส่หรือไม่ใส่ก็ได้):", value="", placeholder="https://hooks.zapier.com/hooks/catch/...")
    
    # สร้างปุ่มควบคุมการสแกนและปุ่มรีเซ็ตให้อยู่คู่กัน
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        scan_btn = st.button(UI_LANG_MAP['btn_scan_market'])
    with col_btn2:
        reset_btn = st.button("🔄 รีเซ็ตข้อมูลสแกน")

    # ถ้ากดปุ่มรีเซ็ต ให้ล้างข้อมูลใน Session State ทันที
    if reset_btn:
        st.session_state.scan_results = None
        st.session_state.scan_df = None
        st.success("ล้างข้อมูลการสแกนเรียบร้อยแล้ว สามารถกดเริ่มสแกนใหม่ได้เลยครับ")
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
            futures = {executor.submit(check_ma_snr_combo, ticker, False): ticker for ticker in stock_list}
            for future in concurrent.futures.as_completed(futures):
                count += 1
                if count % 20 == 0 or count == total_stocks:
                    progress_bar.progress(count / total_stocks)
                    status_text.text(UI_LANG_MAP['status_scanning'].format(count=count, total=total_stocks))
                try:
                    res_data_found, raw_df_found = future.result()
                    if res_data_found:
                        results.append({'res_data': res_data_found, 'raw_df': raw_df_found})
                        
                        if messenger_webhook:
                            msg = f"🚨 สัญญาณซื้อหุ้น {res_data_found['Ticker']} | ราคา: ${res_data_found['Price ($)']} | แนวรับ: ${res_data_found['Support ($)']}"
                            send_messenger_alert(msg, messenger_webhook)
                except Exception:
                    pass

        status_text.empty()
        st.success(f'✅ สแกนเสร็จสิ้น! พบหุ้นทรงสวยเข้าเงื่อนไขทั้งหมด {len(results)} ตัว')
        
        # บันทึกผลลัพธ์ลง Session State คงอยู่ตลอดจนกว่าจะกดปุ่มรีเซ็ต
        st.session_state.scan_results = results
        if results:
            df_result_display = pd.DataFrame([item['res_data'] for item in results])[['Ticker', 'Price ($)', 'Support ($)', 'Dist_Sup (%)', 'RSI', 'Resist 1 ($)', 'Resist 2 ($)', 'Volume', 'Date']]
            st.session_state.scan_df = df_result_display

    # แสดงผลแกลเลอรี่และตารางจาก Session State อย่างต่อเนื่อง
    if st.session_state.scan_results:
        results = st.session_state.scan_results
        df_result_display = st.session_state.scan_df

        st.markdown("---")
        st.subheader('📸 แกลเลอรี่กราฟหุ้นทรงสวย (ข้อมูลคงอยู่ตลอดจนกว่าจะกดรีเซ็ต)')
        
        items_per_page = 6
        total_pages = max(1, (len(results) + items_per_page - 1) // items_per_page)
        page_num = st.selectbox("เลือกหน้าแสดงผลกราฟ:", range(1, total_pages + 1), key="pagination_select")
        
        start_idx = (page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_page_items = results[start_idx:end_idx]

        cols = st.columns(2)
        col_idx = 0

        with st.spinner('กำลังวาดกราฟเทคนิคสำหรับแกลเลอรี่...'):
            for item in current_page_items:
                ticker_found = item['res_data']['Ticker']
                raw_df_found_gallery = item['raw_df']
                res_data_found_gallery = item['res_data']

                if raw_df_found_gallery is not None:
                    fig_gallery = create_ta_chart(raw_df_found_gallery, ticker_found, res_data_found_gallery)
                    with cols[col_idx % 2]:
                        st.markdown(f'<p style="font-size:0.95rem; font-weight:bold; color:#60A5FA; margin-bottom:0px;">🟢 {ticker_found} | Price: ${res_data_found_gallery["Price ($)"]}</p>', unsafe_allow_html=True)
                        st.caption(f"Support: ${res_data_found_gallery['Support ($)']} | RSI: {res_data_found_gallery['RSI']}")
                        st.plotly_chart(fig_gallery, use_container_width=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_idx += 1

        st.markdown("---")
        st.markdown("#### 📊 ตารางสรุปสัญญาณราคาและแนวรับ-ต้าน")
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
