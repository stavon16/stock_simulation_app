# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 21:43:33 2026

@author: 88690
"""
# streamlit run stock_simulation_app.py

import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import streamlit as st
import yfinance as yf

# --- 1. 樣式與配置 ---
# 根據 Streamlit 規範，此指令必須置於除了導入以外的首行
st.set_page_config(layout="wide", page_title="Invest.Log | 總經量化全維度決策系統")


def apply_aesthetic_style():
    """套用自定義 CSS 樣式"""
    st.markdown("""
        <style>
        .stApp { background-color: #FAF9F6; color: #264653; }
        .recommendation-card {
            padding: 20px; border-radius: 12px; background-color: #FFFFFF;
            border: 1px solid #E0E0E0; margin-bottom: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.02);
        }
        .macro-box {
            padding: 20px; border-radius: 12px; background-color: #E9F5F2;
            border-left: 6px solid #2A9D8F; margin-bottom: 25px;
        }
        .countdown-box {
            padding: 15px; border-radius: 10px; background-color: #264653;
            color: white; text-align: center; margin-bottom: 20px;
            font-weight: bold; border-left: 6px solid #E76F51;
        }
        .entry-signal {
            font-size: 1.1rem; font-weight: bold; padding: 5px 12px;
            border-radius: 5px; margin: 8px 0; display: inline-block;
        }
        .price-label { font-size: 1rem; font-weight: bold; margin-bottom: 2px; }
        .buy-price { color: #E76F51; }
        .sell-price { color: #2A9D8F; }
        .metric-small { font-size: 0.85rem; color: #8D99AE; }
        </style>
    """, unsafe_allow_html=True)


# --- 2. 總經與倒數計時工具 ---

def get_next_cpi_date():
    """
    計算下一次美國 CPI 公佈日期（預設為每月 13 號）。
    """
    today = datetime.date.today()
    current_month_cpi = datetime.date(today.year, today.month, 13)
    if today <= current_month_cpi:
        return current_month_cpi
    else:
        # 處理跨年月份
        month = 1 if today.month == 12 else today.month + 1
        year = today.year + 1 if today.month == 12 else today.year
        return datetime.date(year, month, 13)


def show_cpi_countdown():
    """
    在 UI 頂部顯示 CPI 公佈倒數。
    """
    next_date = get_next_cpi_date()
    days_left = (next_date - datetime.date.today()).days
    if days_left == 0:
        st.markdown(
            '<div class="countdown-box">⚠️ 注意：美國 CPI 數據將於今日公佈！</div>',
            unsafe_allow_html=True
        )
    elif days_left <= 3:
        st.markdown(
            f'<div class="countdown-box">🔔 距離美國 CPI 公佈僅剩 {days_left} 天。</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="countdown-box">📊 距離下一次美國 CPI 公佈還有 {days_left} 天</div>',
            unsafe_allow_html=True
        )


@st.cache_data(ttl=86400)
def fetch_macro_data():
    """
    從 FRED 抓取總經數據並判定趨勢。
    """
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=365 * 2)
        end = datetime.datetime.now()
        df = web.DataReader(['CPIAUCSL', 'PPIFIS'], 'fred', start, end)
        df_yoy = df.pct_change(12) * 100
        latest_cpi = df_yoy['CPIAUCSL'].iloc[-1]
        latest_ppi = df_yoy['PPIFIS'].iloc[-1]
        prev_cpi = df_yoy['CPIAUCSL'].iloc[-2]
        status, bias = (
            ("🟢 通膨降溫中", 1.1) if latest_cpi < prev_cpi
            else ("🔴 通膨升溫中", 0.9)
        )
        return latest_cpi, latest_ppi, status, bias
    except Exception as e:
        return 0.0, 0.0, f"數據讀取失敗: {e}", 1.0


# --- 3. 核心量化分析類別 ---

class MultiStockAnalyzer:
    """
    處理股票數據抓取與技術指標計算。
    """

    def __init__(self, tickers, macro_bias=1.0):
        self.tickers = tickers
        self.data = {}
        self.metrics = {}
        self.names = {}
        self.macro_bias = macro_bias

    def load_data(self, period="1y"):
        """抓取 yfinance 數據並計算指標。"""
        for t in self.tickers:
            try:
                stock = yf.Ticker(t)
                self.names[t] = stock.info.get('longName', t)
                df = stock.history(period=period)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    df['Daily_Ret'] = df['Close'].pct_change()
                    # 布林通道與均線
                    df['MA20'] = df['Close'].rolling(20).mean()
                    df['STD20'] = df['Close'].rolling(20).std()
                    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
                    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
                    df['MA5'] = df['Close'].rolling(5).mean()
                    # RSI
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
                    self.data[t] = df
            except Exception:
                continue

    def calculate_metrics(self):
        """依據指標判定信號。"""
        for t, df in self.data.items():
            recent = df.tail(252)
            curr_p = df['Close'].iloc[-1]
            rsi_val = round(df['RSI'].iloc[-1], 1)
            buy_p = round((df['Lower_Band'].iloc[-1] * 0.6) + (df['MA5'].iloc[-1] * 0.4), 2)
            sell_p = round(df['Upper_Band'].iloc[-1], 2)
            
            # 夏普值計算
            vol = recent['Daily_Ret'].std()
            sharpe = (recent['Daily_Ret'].mean() * 252) / (vol * np.sqrt(252)) if vol != 0 else 0
            
            # 信號邏輯
            dist = (curr_p - buy_p) / buy_p
            if rsi_val < 35.0:
                sig, col = " 💎 底部黃金區", "#E76F51"
            elif dist <= 0.02:
                sig, col = " 🔥 買點現蹤", "#E76F51"
            elif rsi_val > 70.0:
                sig, col = " ⚠️ 超漲警戒區", "#264653"
            elif (sell_p - curr_p) / curr_p <= 0.02:
                sig, col = " 🎯 到達賣點", "#2A9D8F"
            else:
                sig, col = " 💤 伺機而動", "#8D99AE"

            self.metrics[t] = {
                '公司名稱': self.names.get(t, t),
                '總報酬率': (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1,
                '夏普值': sharpe,
                '現價': round(curr_p, 2),
                'RSI': rsi_val,
                '建議買價': buy_p,
                '建議賣價': sell_p,
                '預期空間': f"{(sell_p - curr_p) / curr_p:.1%}",
                '信號': sig,
                'Color': col
            }

    def get_matrix(self):
        """產出決策矩陣 DataFrame。"""
        if not self.metrics:
            return pd.DataFrame()
        res = pd.DataFrame(self.metrics).T
        # 評分邏輯
        max_s = res['夏普值'].max() if res['夏普值'].max() > 0 else 1
        max_r = res['總報酬率'].max() if res['總報酬率'].max() > 0 else 1
        res['得分'] = (
            (res['夏普值'] / max_s * 50) + (res['總報酬率'] / max_r * 50)
        ) * self.macro_bias
        return res.sort_values('得分', ascending=False)


# --- 4. 應用程式主介面 ---

def main():
    apply_aesthetic_style()
    st.title("Invest.Log | 總經量化全維度決策系統")
    
    show_cpi_countdown()
    l_cpi, l_ppi, m_status, m_bias = fetch_macro_data()

    with st.sidebar:
        st.header("📊 投資組合")
        tickers_input = st.text_input("輸入代碼 (以逗號分隔)", "2330.TW, 2454.TW, TSLA, NVDA")
        p_map = {"1年": "1y", "3年": "3y", "5年": "5y"}
        sel_p = st.selectbox("分析區間", list(p_map.keys()), index=0)
        run_btn = st.button("啟動分析")

    st.markdown(f"""
        <div class="macro-box">
            <h4>🌍 總體經濟看板</h4>
            最新 CPI: <b>{l_cpi:.2f}%</b> | 最新 PPI: <b>{l_ppi:.2f}%</b><br>
            趨勢判定：{m_status}
        </div>
    """, unsafe_allow_html=True)

    if run_btn:
        ticker_list = [t.strip() for t in tickers_input.split(',')]
        analyzer = MultiStockAnalyzer(ticker_list, m_bias)
        
        with st.spinner("數據分析中，請稍候..."):
            analyzer.load_data(p_map[sel_p])
            analyzer.calculate_metrics()
            df = analyzer.get_matrix()

        if not df.empty:
            st.subheader("🎯 優先推薦標的")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(df.head(3).iterrows()):
                with cols[i % 3]:
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>{row['公司名稱']}</h3>
                            <div class="entry-signal" style="background-color:{row['Color']}22; color:{row['Color']}">
                                {row['信號']}
                            </div><br>
                            建議買價: <b>{row['建議買價']}</b><br>
                            建議賣價: <b>{row['建議賣價']}</b>
                        </div>
                    """, unsafe_allow_html=True)

            st.subheader("📊 核心決策矩陣")
            cols_to_show = ['公司名稱', '得分', '信號', '建議買價', '建議賣價', '現價', 'RSI', '夏普值']
            st.dataframe(
                df[cols_to_show].style.background_gradient(subset=['得分'], cmap='YlGnBu'),
                use_container_width=True
            )

            st.markdown("### 🔍 系統信號快速對照表")
            st.table(pd.DataFrame({
                "信號名稱": ["💎 底部黃金區", "🔥 買點現蹤", "🎯 到達賣點", "⚠️ 超漲警戒區", "💤 伺機而動"],
                "觸發邏輯": ["RSI < 35", "貼近支撐位", "貼近壓力位", "RSI > 70", "常態震盪"],
                "建議動作": ["分批佈局", "高品質進場", "獲利了結", "嚴禁追高", "耐心觀望"]
            }))

        # 深度決策手冊
        with st.expander("📖 深度決策手冊：買賣建議與指標說明"):
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                ### 📥 進場與定價邏輯
                * **建議買價**: 布林下軌 (-2σ) 與 5 日均線加權。
                * **買入判定**: `💎 底部區` (恐慌) 或 `🔥 買點現蹤` (支撐)。
                """)
            with c2:
                st.markdown("""
                ### 📤 出場與定價邏輯
                * **建議賣價**: 以布林上軌 (+2σ) 為目標。
                * **賣出判定**: `⚠️ 超漲區` (過熱) 或 `🎯 到達賣點` (目標)。
                """)
            st.info("💡 操作核心：當標的得分 > 70 且夏普值 > 1 時，若出現買入信號，通常為高品質投資契機。")


if __name__ == "__main__":
    main()