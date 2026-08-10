
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

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="US Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. ฉีด Custom CSS สำหรับตกแต่ง UI หน้าจอมือถือโดยเฉพาะ
st.markdown("""
    <style>
    /* ปรับ Padding รวมของหน้าเว็บ */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* ตกแต่ง Header ให้ขนาดพอดีจอมือถือ */
    .main-title {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    
    .sub-title {
        font-size: 0.95rem !important;
        color: #64748B;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* ขยายปุ่มสแกนให้ใหญ่ สวย เด่น บนมือถือ */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
    }

    /* ซ่อน Footer และ Header เมนูของ Streamlit เพื่อความคลีน */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- ส่วนแสดงผล Header บนเว็บ ---
st.markdown('<div class="main-title">📈 สแกนหุ้นสหรัฐฯ + AI Pattern</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ค้นหาสัญญาณหุ้นกลับตัว และเปรียบเทียบรูปทรงกราฟอัตโนมัติ</div>', unsafe_allow_html=True)


def get_base_directory():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(script_dir):
            return script_dir
    except NameError:
        pass
    return os.getcwd()


def get_us_stock_tickers():
    tickers = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'AMD', 'NFLX',
        'COST', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'INTC', 'CMCSA', 'QCOM', 'TXN', 'AMAT',
        'HON', 'AMGN', 'SBUX', 'BKNG', 'GILD', 'MDLZ', 'ADP', 'ADI', 'VRTX', 'REGN',
        'RKLB', 'IREN', 'EOSE', 'CRWV', 'NUAI', 'WDC', 'PLTR', 'SOUN', 'BBAI', 'IONQ',
        'RGTI', 'QUBT', 'ASTS', 'LUNR', 'JOBY', 'ACHR', 'MARA', 'RIOT', 'CLSK', 'CIFR'
    ]

    try:
        url_nasdaq = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt'
        df_nasdaq = pd.read_csv(url_nasdaq, sep='|')
        nasdaq_stocks = df_nasdaq[
            (df_nasdaq['ETF'] == 'N') & (df_nasdaq['Test Issue'] == 'N')
        ]['Symbol'].tolist()
        tickers.extend(nasdaq_stocks)

        url_other = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt'
        df_other = pd.read_csv(url_other, sep='|')
        other_stocks = df_other[
            (df_other['ETF'] == 'N') & (df_other['Test Issue'] == 'N')
        ]['ACT Symbol'].tolist()
        tickers.extend(other_stocks)
    except Exception:
        pass

    cleaned_tickers = [
        str(t).strip().replace('.', '-')
        for t in tickers
        if isinstance(t, str) and str(t).strip().replace('-', '').isalpha()
    ]
    return list(set(cleaned_tickers))


def check_ma_snr_combo(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='6mo', interval='1d')

        if len(df) < 40 or df['Close'].iloc[-1] < 1.0:
            return None

        df.columns = [col.lower() for col in df.columns]

        fast_ma = df['close'].rolling(window=20).mean()
        slow_ma = df['close'].rolling(window=50).mean()

        lookback_df = df.iloc[-21:-1]
        support_level = lookback_df['low'].min()

        recent_3d_low = df['low'].tail(3).min()
        latest_close = df['close'].iloc[-1]
        latest_open = df['open'].iloc[-1]
        latest_vol = df['volume'].iloc[-1]

        near_support = recent_3d_low <= (support_level * 1.06)
        near_ma50 = recent_3d_low <= (slow_ma.iloc[-1] * 1.03)

        if not (near_support or near_ma50):
            return None

        is_green = (latest_close > latest_open) or (latest_close > df['close'].iloc[-2])
        has_vol = latest_vol >= 300_000

        if is_green and has_vol:
            dist_from_sup = ((latest_close - support_level) / support_level) * 100
            ma_diff_pct = ((fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]) * 100

            return {
                'Ticker': ticker,
                'Price ($)': round(latest_close, 2),
                'Support_Level ($)': round(support_level, 2),
                'Dist_from_Support (%)': f'+{dist_from_sup:.2f}%',
                'MA_Spread (%)': f'+{ma_diff_pct:.2f}%',
                'Volume': f"{latest_vol:,.0f}",
                'Date': df.index[-1].strftime('%Y-%m-%d'),
            }
    except Exception:
        pass
    return None


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

# ปุ่มสแกนแบบเด่นเต็มหน้าจอ
if st.button('🚀 เริ่มสแกนหุ้นทันที'):
    st.info('กำลังเตรียมรายชื่อหุ้นและเริ่มสแกน...')

    stock_list = get_us_stock_tickers()
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    count = 0
    total_stocks = len(stock_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        futures = {executor.submit(check_ma_snr_combo, ticker): ticker for ticker in stock_list}

        for future in concurrent.futures.as_completed(futures):
            count += 1
            progress_bar.progress(count / total_stocks)
            status_text.text(f'สแกนไปแล้ว {count}/{total_stocks} ตัว...')

            res = future.result()
            if res:
                results.append(res)

    st.success(f'✅ สแกนเสร็จสิ้น! พบหุ้นเข้าเงื่อนไขทั้งหมด {len(results)} ตัว')

    if results:
        df_result = pd.DataFrame(results)
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_folder = os.path.join(base_dir, f'Charts_Album_{time_stamp}')
        os.makedirs(chart_folder, exist_ok=True)

        st.subheader('📸 กราฟหุ้นที่พบ (AI Match)')

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        best_scores = []
        best_patterns = []

        # กำหนด Grid: 2 คอลัมน์สำหรับจอมือถือ/คอมฯ เพื่อไม่ให้บีบรูปเล็กเกินไป
        cols = st.columns(2)
        col_idx = 0

        for idx, row in df_result.iterrows():
            ticker = row['Ticker']
            chart_url = f'https://charts2.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l'
            chart_path = os.path.join(chart_folder, f'{ticker}_chart.png')

            try:
                res = requests.get(chart_url, headers=headers, timeout=10)
                if res.status_code == 200:
                    with open(chart_path, 'wb') as f:
                        f.write(res.content)

                    score_str, pattern_name = get_best_pattern_match(chart_path, patterns_folder)
                    best_scores.append(score_str)
                    best_patterns.append(pattern_name)

                    with cols[col_idx % 2]:
                        with st.container():
                            st.image(
                                chart_path,
                                caption=f'🟢 {ticker} | ${row["Price ($)"]} | Match: {score_str}'
                            )
                    col_idx += 1
            except Exception:
                best_scores.append('0.0%')
                best_patterns.append('None')

        df_result['Best_Match_Score'] = best_scores
        df_result['Matched_Pattern'] = best_patterns

        st.subheader('📊 สรุปตารางสัญญาณ')
        st.dataframe(df_result, use_container_width=True)

        csv_bytes = df_result.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label='📥 ดาวน์โหลดไฟล์ CSV',
            data=csv_bytes,
            file_name=f'signals_{time_stamp}.csv',
            mime='text/csv',
        )
