import streamlit as st
import pandas as pd
import yfinance as yf
import finnhub
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# --- 1. CONFIG & REFINED STYLING ---
st.set_page_config(
    page_title="Market Sentiment Intelligence",
    page_icon="https://cdn-icons-png.flaticon.com/512/18220/18220151.png",  # You can use a URL or a local file path here
    layout="wide"
)

st.markdown("""
    <style>
    .justify-text { text-align: justify; line-height: 1.6; }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 20px !important;
        border-radius: 10px;
        height: 120px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .help-box {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 15px;
        border-radius: 5px;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PERSISTENT STATE ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- 3. CORE ENGINE ---
def execute_intelligence_fetch(ticker_input, lookback):
    try:
        # --- SECURE API KEY EXTRACTION ---
        # This pulls from .streamlit/secrets.toml
        SECRET_API_KEY = st.secrets["FINNHUB_API_KEY"]
        
        ticker_obj = yf.Ticker(ticker_input)
        info = ticker_obj.info
        
        # Validate ticker existence
        if not info.get('regularMarketPrice') and not info.get('currentPrice'):
            st.error(f"Invalid Ticker: {ticker_input}")
            return

        client = finnhub.Client(api_key=SECRET_API_KEY)
        fetch_start = datetime.now() - timedelta(days=lookback + 2)
        
        # Fetching news and price data
        news = client.company_news(ticker_input, _from=fetch_start.strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'))
        prices = yf.download(ticker_input, start=fetch_start, interval="5m")

        if len(news) == 0 or prices.empty:
            st.warning("Data gap detected. Try a larger lookback window or a more active ticker.")
            return

        # Load models
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('sentiment_model.pkl')
        
        # Sentiment Processing
        df_news = pd.DataFrame(news)
        X_vec = vectorizer.transform(df_news['headline'])
        df_news['sent_score'] = (model.predict_proba(X_vec)[:, 1] - 0.5) * 2
        df_news['datetime'] = pd.to_datetime(df_news['datetime'], unit='s')
        
        # Cleaning Price Data (Handling Multi-Index from yfinance)
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        prices.index = prices.index.tz_localize(None)
        
        # Merging News with Price Action
        merged = pd.merge_asof(prices.sort_index(), df_news.sort_values('datetime'), 
                              left_index=True, right_on='datetime', direction='backward')
        merged['sent_score'] = merged['sent_score'].fillna(0)
        merged['str_time'] = merged.index.strftime('%b %d, %H:%M')

        # Saving to state
        st.session_state.master_df = merged
        st.session_state.news_df = df_news
        st.session_state.ticker_metadata = info
        st.session_state.data_loaded = True
        
        # Refresh to display dashboard
        st.rerun()

    except Exception as e:
        st.error(f"Analytical Engine Failure: {e}")

# --- 4. MAIN INTERFACE (ALWAYS VISIBLE AT TOP) ---
st.title("Market Intelligence Terminal")
st.subheader("Analysis Parameters")

col1, col2 = st.columns(2)
with col1:
    ticker_input = st.text_input("Target Ticker Symbol", value="MU").upper()
with col2:
    lookback = st.select_slider("Historical Window (Days)", options=[1, 3, 5, 7, 10, 14], value=7)

st.write("")

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    if st.button("Generate Executive Report", use_container_width=True, type="primary"):
        with st.spinner(f"Analyzing {ticker_input}..."):
            execute_intelligence_fetch(ticker_input, lookback)

st.markdown("---")

# --- 5. UI RENDERING ---

if not st.session_state.data_loaded:
    st.header("Market Intelligence Framework: A Deep Dive into Data-Driven Sentiment Analysis")
    
    st.subheader("The Foundation: Neural Sentiment Mapping")
    st.write("""
    The backbone of this terminal is a sophisticated Natural Language Processing (NLP) pipeline designed 
    to bridge the gap between human emotion and market movement. The core sentiment engine was pre-trained 
    on a massive dataset of IMDB movie reviews. This specific training set was chosen because it contains 
    a high density of polarized, expressive language that allows the model to detect subtle emotional 
    cues in financial text. To process this data, we utilized **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization**. 
    This method assigns a mathematical weight to words based on their unique impact; for example, common 
    filler words are ignored, while high-stakes terms like 'Lawsuit,' 'Breakthrough,' or 'Earnings' 
    are prioritized as key drivers of volatility.
    """)
    st.image("sentence break.png", caption="The process of converting a raw text sentence into a structured vector of numbers (embedding)")

    st.subheader("Quantifying Market Regime with the Hurst Exponent")
    st.write("""
    To distinguish between a true market trend and random price noise, the terminal calculates the 
    **Hurst Exponent ($H$)**. This mathematical indicator measures the 'memory' of a price series to 
    determine the current market regime. A Hurst value above **0.50** indicates a **Trending Regime**, 
    where price movements are likely to persist in their current direction. Conversely, a value below 
    **0.50** signals a **Mean-Reverting Regime**, suggesting the price is 'choppy' and likely to bounce 
    back to its average. Understanding this regime is critical, as sentiment-based signals are far 
    more reliable in trending markets than in random, noisy ones.
    """)
    st.write("""
             How to Interpret the Regime:
             * H > 0.5 — The Highway (Trending): This person is walking with a destination in mind. If their last 10 steps were forward, their next step is statistically likely to be forward. The market has **Long-term Memory**. 
             Strategic Play:* If $H = 0.58$ and you see a **Green Sentiment Bubble**, you can have high confidence the trend will persist.

             * H < 0.5 — The Bungee Cord (Mean-Reverting): This person is tied to a pole with a 5-meter bungee cord. They can run forward, but the further they go, the harder they get pulled back to the center. This is **Negative Memory**.
             Strategic Play:* If $H = 0.35$ and you see a **Green Sentiment Bubble**, be careful! The price is likely to 'snap back' to its average immediately.
             
             * H = 0.5 — The Random Walk: This is a drunk person. Their next step is totally random and unpredictable. """)


    st.subheader("Narrative Bias and Price Sensitivity")
    st.write("""
    Beyond individual headlines, the system monitors **Narrative Bias**, which is the average emotional 
    score of the entire news cycle for a specific ticker. This acts as a macro-level barometer for 
    the stock's 'vibe'. Parallel to this, we measure **Sensitivity (Narrative Alignment)** through 
    Pearson Correlation. This metric quantifies the 'glue' between news and price action. If a stock 
    shows a high sensitivity percentage (e.g., **51%**), it confirms that the asset is news-driven. 
    If sensitivity is low, it warns the analyst that the price is likely being moved by external 
    macro factors, such as interest rates, rather than company-specific headlines.
    """)

    st.subheader("Visualizing Alpha: The Narrative-Price Fusion")
    st.write("""
    The terminal's primary diagnostic tool is the **Narrative-Price Fusion** chart. By utilizing a 
    **Categorical X-Axis**, we 'stitch' trading sessions together, removing the blank spaces caused 
    by nights and weekends. This creates a continuous, gap-free line that makes it easy to spot direct 
    cause-and-effect. On this chart, sentiment 'bubbles' are plotted directly over price movements. 
    The **size** of the bubble represents the conviction of the sentiment, while the **color** (Red/Green) 
    identifies the direction. This allows an analyst to instantly see if a major news event successfully 
    triggered a price surge or if the market ignored the news entirely.
    """)

    st.image("NB1.png")
    st.image("NB2.png", caption="Decoding the Narrative-Price Fusion Chart")

    st.write("""
    ### **How to Read the Visual Signals**
    The visual architecture of the fusion chart relies on two primary markers: bubble size and color scale. 
    The size of each bubble represents **Conviction**, which is determined by the intensity of the sentiment score. 
    A large bubble, like the clusters seen in the center of the chart, indicates a 'High-Conviction' news event 
    where the AI detected powerful, market-moving language—such as 'Record Revenue' or 'Major Partnership'—rather 
    than generic corporate filler. Meanwhile, the color indicates the **Mood** or direction of the news. 
    Bright green bubbles represent highly positive catalysts (+0.8 to +1.0), while orange and red bubbles 
    signal negative pressure. The neutral yellow bubbles represent 'Market Noise' or headlines that are 
    mathematically unlikely to move the needle. When you see a cluster of large green bubbles appearing 
    just as the blue price line begins a steep upward slope, you have found the **Fusion Point**. 
    This is the ultimate 'Trade Signal' where the news narrative has successfully fused with the price, 
    providing the fundamental fuel necessary for a sustainable trend.
    """)

    st.write("""
    ### **Strategic Cautions: The Divergence Trap**
    When analyzing this graph, it is critical to maintain a disciplined watch for **Sentiment Exhaustion**. 
    If you observe a series of large green bubbles appearing while the blue price line stays flat or 
    starts to drop, you are looking at a **Divergence**. This suggests that the 'Good News' is already 
    priced in and buyers are exhausted, often signaling an imminent reversal. Furthermore, if the 
    **Hurst Exponent** is low (below 0.5), any sudden price spike following a bubble is likely a 
    'Bungee Cord' move—a temporary jump that lacks the 'memory' to sustain itself and will quickly 
    snap back to the average. As a rule of professional caution, you should always look for the price 
    to 'hold' its gains after a significant news cluster before confirming a long-term entry.
    """) 

    st.subheader("The Strategy Growth Narrative")
    st.write("""
    To provide a real-world perspective on performance, the system tracks a **Hypothetical $100 Strategy Growth** line. This transforms abstract percentage points into a tangible dollar value, showing exactly how 
    an initial investment would have evolved over the lookback window. This visual is supported by a 
    **Strategic Word Cloud** that uses strict filtering to remove generic corporate 'filler'. By stripping 
    away words like the ticker name or 'Stock,' the cloud reveals the actual thematic drivers—such as 
    **'AI,' 'Memory,'** or **'Revenue'**—that are fueling the bubbles on the main chart.
    """)

    st.divider()
    st.subheader("Conclusion: Navigating the Intersection of News and Price")
    st.write("""
    The Market Intelligence Terminal is designed to shift the analytical focus from 'what' the price 
    is to '**why**' the price is moving. By combining deep-learning sentiment analysis with rigorous 
    statistical indicators like the Hurst Exponent and Narrative Sensitivity, it filters the noise 
    of the 24-hour news cycle into a clear, actionable signal. However, successful operation requires 
    caution: users must account for sentiment lag and be wary of 'random walk' regimes where the 
    Hurst value sits near 0.50.
    """)

else:
    merged = st.session_state.master_df
    info = st.session_state.ticker_metadata
    df_news = st.session_state.news_df

    # HEADER & SUMMARY
    st.header(f"{info.get('longName', ticker_input)}")
    st.markdown(f'<div class="justify-text">{info.get("longBusinessSummary", "N/A")}</div>', unsafe_allow_html=True)
    st.divider()

    # SECTION 1: PERFORMANCE SCORECARD & NARRATIVE PIE
    col_scores, col_pie = st.columns([2, 1])

    with col_scores:
        st.subheader("Performance Indicators")
        m1, m2, m3, m4 = st.columns(4)
        
        # Calculations
        current_price = merged['Close'].iloc[-1]
        prev_close = info.get('previousClose', current_price)
        day_change = ((current_price - prev_close) / prev_close) * 100
        
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(merged['Close'].values[lag:], merged['Close'].values[:-lag]))) for lag in lags]
        h_val = np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
        
        bias = merged['sent_score'].mean()
        sensitivity = merged[['Close', 'sent_score']].corr().iloc[0,1]

        m1.metric("Live Price", f"${current_price:.2f}", f"{day_change:+.2f}%")
        m2.metric("Regime (Hurst)", f"{h_val:.3f}")
        m3.metric("Narrative Bias", f"{bias:+.2f}")
        m4.metric("Sensitivity", f"{abs(sensitivity):.2%}")

        with st.expander("ℹ️ Metric Documentation"):
            st.markdown("""
            * **Live Price:** Current traded price and percentage change relative to previous close.
            * **Hurst Exponent:** Measures 'memory' in price. >0.5 is trending, <0.5 is mean-reverting.
            * **Narrative Bias:** Average sentiment score of all news (-1 to +1).
            * **Sensitivity:** Correlation between sentiment shifts and price action.
            """)

    with col_pie:
        sent_cat = pd.cut(df_news['sent_score'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])
        pie_fig = px.pie(names=sent_cat.value_counts().index, values=sent_cat.value_counts().values, 
                         title="Sentiment Composition", hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
        pie_fig.update_layout(height=280, margin=dict(t=50, b=0, l=0, r=0), showlegend=True)
        st.plotly_chart(pie_fig, use_container_width=True)

    # SECTION 2: ROI GROWTH CHART (FULL WIDTH)
    st.divider()
    st.subheader("Strategy ROI Growth ($100 Base)")
    
    price_multiples = merged['Close'] / merged['Close'].iloc[0]
    merged['strategy_growth'] = 100 * price_multiples
    
    growth_fig = go.Figure()
    growth_fig.add_trace(go.Scatter(
        x=merged['str_time'], y=merged['strategy_growth'],
        mode='lines', line=dict(color='#00C805', width=3),
        fill='tozeroy', fillcolor='rgba(0, 200, 5, 0.1)',
        hovertemplate='Account Value: $%{y:.2f}<extra></extra>'
    ))

    unique_dates = merged.index.strftime('%b %d').unique()
    tick_vals = [merged[merged.index.strftime('%b %d') == d]['str_time'].iloc[0] for d in unique_dates]
    
    growth_fig.update_xaxes(type='category', tickmode='array', tickvals=tick_vals, ticktext=unique_dates, gridcolor='rgba(128,128,128,0.1)', tickangle=0)
    growth_fig.update_yaxes(gridcolor='rgba(128,128,128,0.1)', tickprefix="$")
    growth_fig.update_layout(height=400, margin=dict(t=10, b=50, l=10, r=10), template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(growth_fig, use_container_width=True)

    # SECTION 3: FUSION CHART
    st.divider()
    st.subheader("Narrative-Price Fusion")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged['str_time'], y=merged['Close'], mode='lines', line=dict(color='#1A73E8', width=2), hoverinfo='skip'))
    
    events = merged[merged['sent_score'].abs() > 0.2].copy()
    fig.add_trace(go.Scatter(
        x=events['str_time'], y=events['Close'], mode='markers',
        marker=dict(size=events['sent_score'].abs()*25+5, color=events['sent_score'], 
                    colorscale='RdYlGn', showscale=True, colorbar=dict(orientation='h', y=-0.25, thickness=15)),
        text=events['headline'],
    ))

    fig.update_xaxes(type='category', tickmode='array', tickvals=tick_vals, ticktext=unique_dates, gridcolor='rgba(128,128,128,0.1)', tickangle=0)
    fig.update_layout(height=600, margin=dict(t=10, b=150, l=50, r=10), template="plotly_dark", showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # SECTION 4: WORD CLOUD
    st.divider()
    st.subheader("Narrative Sentiment Cloud")
    
    custom_stops = set(STOPWORDS)
    name_parts = info.get('longName', "").lower().split()
    custom_stops.update([ticker_input.lower(), "stock", "stocks", "market", "shares", "price", "inc", "corp", "technology"] + name_parts)
    
    text = " ".join(headline for headline in df_news.headline)
    wc = WordCloud(stopwords=custom_stops, width=1600, height=400, background_color=None, mode="RGBA").generate(text)
    
    fig_wc, ax = plt.subplots(figsize=(16, 4))
    fig_wc.patch.set_alpha(0) 
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig_wc)
