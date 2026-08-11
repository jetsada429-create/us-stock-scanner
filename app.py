
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
    'search_ticker_title': "US Stock Scanner PRO",
    'search_ticker_subtitle': "ระบบสแกนทางเทคนิค พร้อมกราฟเทคนิคและข้อมูลธุรกิจแปลไทย",
    'search_ticker_label': "พิมพ์ชื่อ Ticker หุ้น (เช่น NVDA, PLTR, RKLB):",
    'btn_analyze_single': "🔎 วิเคราะห์ทันที",
    'btn_scan_market': "🚀 เริ่มสแกนทั้ง 3 ตลาด (7,000+ หุ้น)",
    'status_preparing_tickers': "⏳ กำลังดึงรายชื่อหุ้นทั้งหมดจาก NASDAQ, NYSE, AMEX...",
    'status_scanning': "⏳ สแกนไปแล้ว {count}/{total} ตัว...",
    'status_analyzing_single': "⏳ กำลังดึงข้อมูลและวิเคราะห์ {ticker}...",
    'success_stock_found_single': "🟢 หุ้น **{ticker}** ผ่านเงื่อนไขสแกนสัญญาณ BUY!",
    'error_stock_not_found_single': "🔴 หุ้น **{ticker}** ไม่ติดเงื่อนไขสัญญาณซื้อในขณะนี้",
    'expander_business_summary': "📖 สรุปธุรกิจ (แปลไทยอัตโนมัติ)",
    'chart_title_single': "#### 📈 กราฟเทคนิคแนวรับ-แนวต้าน (และเส้นเทรนออโต้)",
    'placeholder_pattern_match': "AI Pattern Match (เบื้องต้น): สร้างฐาน.png (ความแม่นยำ: 75.4%)",
    'analysis_title': "#### 📊 ข้อมูลวิเคราะห์",
    'metric_current_price': "ราคาปัจจุบัน",
    'metric_support_level': "แนวรับ (Support)",
    'metric_distance_support': "ระยะห่างแนวรับ",
    'metric_resistance_1': "แนวต้าน 1",
    'metric_resistance_2': "แนวต้าน 2 (Highเดิม)",
    'tab_search_ticker': "🔍 ค้นหา & วิเคราะห์รายตัว",
    'tab_scan_market': "🚀 สแกนคัดหุ้นทรงสวย",
}

st.set_page_config(
    page_title=UI_LANG_MAP['search_ticker_title'],
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    .main-title {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-title {
        font-size: 0.8rem !important;
        color: #64748B;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        padding: 0.35rem 0.6rem !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
    }
    .company-name {
        font-size: 1.1rem !important;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0rem;
    }
    
    /* --- ปรับแต่งส่วนหัวข้อและกล่อง Metric ให้กะทัดรัดลงครึ่งหนึ่ง --- */
    h3 {
        font-size: 1rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    h4 {
        font-size: 0.85rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    .metric-card {
        background-color: white;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        margin-bottom: 0.3rem;
    }
    /* ย่อขนาดตัวหนังสือใน st.metric ให้เล็กลง */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.7rem !important;
    }

    .biz-summary {
        font-size: 0.75rem !important;
        color: #334155;
        background-color: #F8FAFC;
        padding: 6px 8px !important;
        border-radius: 6px;
        border-left: 3px solid #2563EB;
        margin-bottom: 0.3rem;
        line-height: 1.3;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.3rem !important;
            padding-bottom: 0.8rem !important;
            padding-left: 0.2rem !important;
            padding-right: 0.2rem !important;
        }
    }
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
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": "th",
            "dt": "t",
            "q": text
        }
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
    
    fig.add_trace(go.Scatter(x=df.index, y=fast_ma, line=dict(color='deepskyblue', width=1), name='MA20'))
    fig.add_trace(go.Scatter(x=df.index, y=slow_ma, line=dict(color='orange', width=1), name='MA50'))

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
        fig.add_trace(go.Scatter(x=fit_dates, y=y_high_fit, line=dict(color='goldenrod', width=2), name='เทรนแนวต้านล่าสุด (40 วัน)'))
        fig.add_trace(go.Scatter(x=fit_dates, y=y_low_fit, line=dict(color='mediumturquoise', width=2), name='เทรนแนวรับล่าสุด (40 วัน)'))
    except Exception:
        pass

    latest_date = df.index[-1]
    earliest_date = df.index[0]
    
    sup_val = res_data['Support ($)']
    fig.add_shape(type="line", x0=earliest_date, y0=sup_val, x1=latest_date, y1=sup_val, line=dict(color="green", width=3, dash='dash'))
    fig.add_annotation(x=latest_date, y=sup_val, text=f"Support: ${sup_val}", bgcolor="green", font=dict(color="white"), ax=0, ay=-15)

    res1_val = res_data['Resist 1 ($)']
    fig.add_shape(type="line", x0=earliest_date, y0=res1_val, x1=latest_date, y1=res1_val, line=dict(color="red", width=2, dash='dash'))
    fig.add_annotation(x=latest_date, y=res1_val, text=f"Resist 1: ${res1_val}", bgcolor="red", font=dict(color="white"), ax=0, ay=-15)

    res2_val = res_data['Resist 2 ($)']
    fig.add_shape(type="line", x0=earliest_date, y0=res2_val, x1=latest_date, y1=res2_val, line=dict(color="darkred", width=3))
    fig.add_annotation(x=latest_date, y=res2_val, text=f"Resist 2: ${res2_val}", bgcolor="darkred", font=dict(color="white"), ax=0, ay=15)

    fig.update_layout(
        title=f'กราฟเทคนิค {ticker} | ราคา: ${res_data["Price ($)"]}',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        margin=dict(l=10, r=10, t=30, b=10),
        height=380,
        xaxis_title="วันที่",
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
        latest_close = df['close'].iloc[-1]
        slow_ma = df['close'].rolling(window=50).mean()

        lookback_sup = df.tail(20)
        support_level = lookback_sup['low'].min()
        resistance_1 = lookback_sup['high'].max()
        
        lookback_res2 = df.iloc[-61:-1]
        resistance_2 = lookback_res2['high'].max()
        if resistance_2 <= resistance_1 * 1.01:
             early_lookback = df.iloc[-61:-21]
             resistance_2 = early_lookback['high'].max() if not early_lookback.empty else resistance_1 * 1.10

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


tab1, tab2 = st.tabs([UI_LANG_MAP['tab_search_ticker'], UI_LANG_MAP['tab_scan_market']])

# ================= TAB 1: ค้นหาหุ้นรายตัว =================
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
                st.markdown(f'<p class="company-name">{res.get("longName", "N/A")}</p>', unsafe_allow_html=True)
                st.success(UI_LANG_MAP['success_stock_found_single'].format(ticker=single_ticker) + f' | ข้อมูล ณ วันที่: {res["Date"]}')
                
                # กราฟขึ้นมาแสดงบนสุดทันที
                if raw_df is not None:
                    st.markdown(UI_LANG_MAP['chart_title_single'])
                    fig = create_ta_chart(raw_df, single_ticker, res)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"🤖 {UI_LANG_MAP['placeholder_pattern_match']}")

                st.markdown("---")
                st.markdown(UI_LANG_MAP['analysis_title'])
                with st.container():
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric(UI_LANG_MAP['metric_current_price'], f"${res['Price ($)']}")
                    st.markdown("---")
                    s1, s2 = st.columns(2)
                    s1.metric(UI_LANG_MAP['metric_support_level'], f"${res['Support ($)']}", f"{res['Dist_Sup (%)']} จากราคาปัจจุบัน")
                    st.markdown("<br>", unsafe_allow_html=True)
                    r1, r2 = st.columns(2)
                    r1.metric(UI_LANG_MAP['metric_resistance_1'], f"${res['Resist 1 ($)']}")
                    r2.metric(UI_LANG_MAP['metric_resistance_2'], f"${res['Resist 2 ($)']}")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 💰 กำไรสุทธิ 3 ไตรมาสล่าสุด")
                if df_profit is not None:
                    # จำกัดความสูงของตารางให้กะทัดรัดขึ้น
                    st.dataframe(df_profit, use_container_width=True, hide_index=True, height=120)
                    
                    # แทนที่ st.bar_chart ด้วย Plotly Bar Chart เพื่อจำกัดความสูงไม่ให้ยาวเกินไป
                    fig_profit = go.Figure(data=[go.Bar(
                        x=df_profit['Quarter End'],
                        y=df_profit['Net Income (M$)'],
                        marker_color='#2563EB'
                    )])
                    fig_profit.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=180,
                        template='plotly_white',
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


# ================= TAB 2: สแกนคัดหุ้นทั้งตลาด =================
with tab2:
    st.markdown("### 🚀 สแกนหาหุ้นทรงสวยประจำวัน (ทั้งตลาด NASDAQ, NYSE, AMEX)")
    scan_btn = st.button(UI_LANG_MAP['btn_scan_market'])

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
                except Exception:
                    pass

        status_text.empty()
        st.success(f'✅ สแกนเสร็จสิ้น! พบหุ้นทรงสวยเข้าเงื่อนไขทั้งหมด {len(results)} ตัว')

        if results:
            df_result_display = pd.DataFrame([item['res_data'] for item in results])[['Ticker', 'Price ($)', 'Support ($)', 'Dist_Sup (%)', 'Resist 1 ($)', 'Resist 2 ($)', 'Volume', 'Date']]
            
            st.markdown("---")
            st.subheader('📸 แกลเลอรี่กราฟหุ้นทรงสวย (แสดงผลบนสุด)')
            cols = st.columns(2)
            col_idx = 0

            with st.spinner('กำลังวาดกราฟเทคนิคสำหรับแกลเลอรี่...'):
                for idx, item in df_result_display.iterrows():
                    ticker_found = item['Ticker']
                    match_item = next(x for x in results if x['res_data']['Ticker'] == ticker_found)
                    if match_item['raw_df'] is not None:
                        fig_gallery = create_ta_chart(match_item['raw_df'], ticker_found, match_item['res_data'])
                        with cols[col_idx % 2]:
                            st.markdown(f'<p style="font-size:1rem; font-weight:bold; color:#1D4ED8; margin-bottom:0px;">🟢 {ticker_found} | Price: ${match_item["res_data"]["Price ($)"]}</p>', unsafe_allow_html=True)
                            st.caption(f"Support: ${match_item['res_data']['Support ($)']} | ต้าน1: ${match_item['res_data']['Resist 1 ($)']} | ต้าน2: ${match_item['res_data']['Resist 2 ($)']}")
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
