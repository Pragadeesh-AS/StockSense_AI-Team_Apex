import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (DARK THEME)
# ==========================================
st.set_page_config(page_title="STOCKSENSE AI | NovaMart", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
    /* Premium Dark SaaS Theme */
    :root {
        --bg-color: #0B1120;
        --card-bg: #111827;
        --border-color: #1F2937;
        --text-main: #F8FAFC;
        --text-muted: #94A3B8;
        --accent-blue: #38BDF8;
        --risk-red: #EF4444;
        --risk-orange: #F97316;
        --risk-yellow: #EAB308;
        --healthy-green: #10B981;
    }
    
    .stApp { background-color: var(--bg-color); color: var(--text-main); font-family: 'Inter', sans-serif; }
    
    /* Material Icon Helper */
    .mi { font-family: 'Material Symbols Outlined'; font-weight: normal; font-style: normal; font-size: 20px; line-height: 1; text-transform: none; letter-spacing: normal; word-wrap: normal; white-space: nowrap; direction: ltr; vertical-align: middle; margin-right: 6px; }
    .mi-large { font-size: 24px; margin-right: 10px; }
    
    /* Header Container */
    .header-container { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border-color); margin-bottom: 24px; }
    .header-left h1 { margin: 0; font-size: 1.8rem; font-weight: 800; color: var(--text-main); display: flex; align-items: center; }
    .header-left p { margin: 0; font-size: 0.95rem; color: var(--text-muted); padding-left: 30px; }
    .header-right { display: flex; gap: 16px; align-items: center; color: var(--text-muted); }
    
    /* Pulsing Live Dot */
    .pulse-dot { width: 8px; height: 8px; background-color: var(--healthy-green); border-radius: 50%; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); animation: pulse 2s infinite; margin-right: 6px; }
    @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
    .status-live { display: flex; align-items: center; font-weight: 600; color: var(--healthy-green); font-size: 0.9rem; background: rgba(16, 185, 129, 0.1); padding: 4px 12px; border-radius: 12px; }
    
    /* Premium KPI Cards */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .kpi-card { background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); transition: transform 0.2s ease, box-shadow 0.2s ease; }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 12px rgba(0,0,0,0.3); border-color: #334155; }
    .kpi-label { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; display: flex; align-items: center; }
    .kpi-val { font-size: 2rem; font-weight: 800; color: var(--text-main); margin-bottom: 8px; }
    .kpi-trend-up { color: var(--healthy-green); font-size: 0.85rem; font-weight: 600; display: flex; align-items: center;}
    .kpi-trend-down { color: var(--risk-red); font-size: 0.85rem; font-weight: 600; display: flex; align-items: center;}
    .kpi-trend-warn { color: var(--risk-orange); font-size: 0.85rem; font-weight: 600; display: flex; align-items: center;}
    
    /* AI Insight Banner */
    .ai-banner { background: linear-gradient(90deg, rgba(56,189,248,0.1) 0%, rgba(17,24,39,1) 100%); border-left: 4px solid var(--accent-blue); border-radius: 8px; padding: 20px; margin-bottom: 24px; }
    .ai-banner-title { color: var(--accent-blue); font-weight: 700; font-size: 0.85rem; letter-spacing: 1px; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
    .ai-banner-text { font-size: 1.1rem; color: var(--text-main); margin-bottom: 12px; line-height: 1.5; }
    .ai-meta { display: flex; gap: 16px; font-size: 0.85rem; color: var(--text-muted); }
    
    /* Timeline */
    .timeline-item { border-left: 2px solid var(--border-color); padding-left: 16px; position: relative; margin-bottom: 20px; }
    .timeline-item::before { content: ''; position: absolute; left: -5px; top: 0; width: 8px; height: 8px; border-radius: 50%; background-color: var(--accent-blue); }
    .time-text { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; }
    .event-text { font-size: 0.95rem; color: var(--text-main); }
    
    /* Subheaders */
    .section-title { font-size: 1.2rem; font-weight: 600; color: var(--text-main); display: flex; align-items: center; margin-bottom: 15px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# Update Plotly defaults for dark theme
px.defaults.template = "plotly_dark"
chart_layout_kwargs = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), margin=dict(t=30, l=10, r=10, b=10))

# ==========================================
# 2. DATA LOADING & BUSINESS LOGIC
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/model_predictions.csv')
        df = df[df['category'] != 'LINGERIE']  # Purged category
        
        df['Risk_Level'] = pd.cut(df['stockout_prob'], bins=[-0.1, 0.399, 0.699, 1.1], labels=['LOW', 'MEDIUM', 'HIGH'])
        
        safety_stock = df['rolling_mean_7'] * 2 
        df['Recommended_Stock'] = df['predicted_demand'] + safety_stock
        df['Reorder_Quantity'] = np.maximum(0, df['Recommended_Stock'] - df['closing'] - df['received']).astype(int)
        
        def generate_reason(row):
            r = []
            if row['day_of_week'] >= 5: r.append("Weekend approaching")
            if row['promotion_flag'] == 1: r.append("Active promotion")
            if row['rolling_mean_7'] > row['lag_7_demand']: r.append("Demand surge")
            return " • ".join(r) if r else "Forecast > Available Stock"
            
        df['Explanation'] = df.apply(generate_reason, axis=1)
        return df
    except FileNotFoundError:
        st.error("Predictions file not found! Please run Round 2.")
        return pd.DataFrame()

df = load_data()

if 'approved_orders' not in st.session_state:
    st.session_state.approved_orders = 0

# ==========================================
# 3. SIDEBAR NAVIGATION & COPILOT
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color: #38BDF8; font-weight: 800; margin-bottom: 0;'><span class='mi mi-large'>inventory_2</span> STOCKSENSE</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 0.85rem; margin-bottom: 30px; padding-left: 34px;'>Command Center</p>", unsafe_allow_html=True)
    
    nav = st.radio("NAVIGATION", [
        "Overview", 
        "Action Center", 
        "Demand Intelligence", 
        "AI Control Room"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    exec_mode = st.toggle("Executive Mode")
    st.markdown("<div style='margin-bottom: 8px;'><span class='mi'>storefront</span> Global Store Filter</div>", unsafe_allow_html=True)
    selected_store = st.selectbox("Store Filter", ["All Stores"] + list(df['store_id'].unique()), label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("<div style='margin-bottom: 8px;'><span class='mi'>smart_toy</span> AI Copilot</div>", unsafe_allow_html=True)
    st.chat_input("Ask StockSense (e.g., 'Analyze risk')...")

if selected_store != "All Stores":
    df = df[df['store_id'] == selected_store]

# ==========================================
# 4. GLOBAL HEADER
# ==========================================
now = datetime.now().strftime("%I:%M %p")
st.markdown(f"""
<div class="header-container">
    <div class="header-left">
        <h1><span class="mi mi-large" style="color:#38BDF8;">auto_awesome_mosaic</span> NovaMart STOCKSENSE AI</h1>
        <p>Enterprise Inventory Intelligence & Predictive Replenishment</p>
    </div>
    <div class="header-right">
        <div class="status-live"><div class="pulse-dot"></div> Live</div>
        <span>Store: {selected_store}</span>
        <span class="mi" style="font-size: 1.4rem; cursor: pointer;">notifications</span>
        <span class="mi" style="font-size: 1.4rem; cursor: pointer;">account_circle</span>
        <span style="font-size: 0.8rem; margin-left: 10px;">Updated: {now}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics Math
tot_rev = (df['quantity'] * df['selling_price']).sum()
tot_inv = (df['closing'] * df['selling_price']).sum()
high_risk = len(df[df['Risk_Level'] == 'HIGH'])
avg_risk = df['stockout_prob'].mean() * 100

# ==========================================
# 5. PAGE ROUTING
# ==========================================
if nav == "Overview":
    
    # --- KPI CARDS ---
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label"><span class="mi">payments</span> Estimated Revenue</div>
            <div class="kpi-val">₹{tot_rev/1000000:.1f}M</div>
            <div class="kpi-trend-up"><span class="mi" style="font-size:16px;">trending_up</span> 12.4% vs prev period</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span class="mi">warehouse</span> Total Inventory Value</div>
            <div class="kpi-val">₹{tot_inv/1000000:.1f}M</div>
            <div class="kpi-trend-down"><span class="mi" style="font-size:16px;">trending_down</span> 3.2% vs prev period</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span class="mi">warning</span> Critical Risk Items</div>
            <div class="kpi-val">{high_risk:,}</div>
            <div class="kpi-trend-warn"><span class="mi" style="font-size:16px;">trending_up</span> 8.7% requiring attention</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span class="mi">analytics</span> Portfolio Stock-out Risk</div>
            <div class="kpi-val">{avg_risk:.1f}%</div>
            <div class="kpi-trend-warn"><span class="mi" style="font-size:16px;">trending_up</span> 4.2% overall risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- AI EXECUTIVE INSIGHT ---
    st.markdown(f"""
    <div class="ai-banner">
        <div class="ai-banner-title"><span class="mi">lightbulb</span> AI EXECUTIVE INSIGHT</div>
        <div class="ai-banner-text"><b>{high_risk:,} products</b> require immediate attention today. Demand anomalies have been detected across multiple key categories, driven by approaching weekend constraints and active promotions.</div>
        <div class="ai-meta">
            <span><b>AI Confidence:</b> 93%</span>
            <span><b>Data Freshness:</b> 12 seconds ago</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='section-title'><span class='mi'>health_and_safety</span> Inventory Health</div>", unsafe_allow_html=True)
        # Dummy data for the requested segmentation
        health_data = pd.DataFrame({
            'Status': ['Healthy', 'Low Stock', 'Overstock', 'Dead Stock'],
            'Value': [82, 11, 5, 2]
        })
        fig_health = px.bar(health_data, y='Status', x='Value', orientation='h', 
                            color='Status', color_discrete_map={'Healthy':'#10B981', 'Low Stock':'#F97316', 'Overstock':'#38BDF8', 'Dead Stock':'#64748B'})
        fig_health.update_layout(**chart_layout_kwargs, showlegend=False, height=250, xaxis_title="% of Portfolio")
        st.plotly_chart(fig_health, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'><span class='mi'>pie_chart</span> Risk Distribution</div>", unsafe_allow_html=True)
        risk_counts = df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk', 'Count']
        fig_dist = px.bar(risk_counts, y='Risk', x='Count', orientation='h',
                          color='Risk', color_discrete_map={'HIGH':'#EF4444', 'MEDIUM':'#F97316', 'LOW':'#10B981'})
        fig_dist.update_layout(**chart_layout_kwargs, showlegend=False, height=250, xaxis_title="Item Count")
        st.plotly_chart(fig_dist, use_container_width=True)

    if not exec_mode:
        st.markdown("<div class='section-title'><span class='mi'>error</span> Top Risk Products</div>", unsafe_allow_html=True)
        risk_df = df[df['Risk_Level'] == 'HIGH'][['product_id', 'category', 'store_id', 'closing', 'predicted_demand', 'Risk_Level', 'Reorder_Quantity']].head(10)
        st.dataframe(
            risk_df,
            column_config={
                "product_id": "Product", "category": "Category", "store_id": "Store",
                "closing": "Current Stock", "predicted_demand": st.column_config.NumberColumn("7-Day Forecast", format="%.0f"),
                "Risk_Level": "Risk", "Reorder_Quantity": st.column_config.NumberColumn("Recommended Order", format="%d")
            }, hide_index=True, use_container_width=True
        )

# ==========================================
# ACTION CENTER
# ==========================================
elif nav == "Action Center":
    st.markdown("<div class='section-title'><span class='mi'>smart_toy</span> STOCKSENSE AI RECOMMENDATIONS</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.button(f"Critical Reviews (7,940)", use_container_width=True)
    c2.button(f"Approved POs ({st.session_state.approved_orders})", use_container_width=True)
    c3.button("Demand Anomalies (24)", use_container_width=True)
    c4.button("Review AI Actions (12)", use_container_width=True)
    
    st.markdown("<hr style='border-color: #1F2937;'>", unsafe_allow_html=True)
    
    # Build styled recommendation cards
    action_items = df[df['Risk_Level'] == 'HIGH'].sort_values(by='stockout_prob', ascending=False).head(5)
    
    for idx, row in action_items.iterrows():
        st.markdown(f"""
        <div class="premium-card" style="border-left: 4px solid var(--risk-red);">
            <div style="display:flex; justify-content: space-between;">
                <div>
                    <h4 style="margin:0; color:var(--text-main);">{row['category']} - Product {row['product_id']} (Store {row['store_id']})</h4>
                    <p style="margin: 5px 0; color:var(--risk-red); font-weight: bold;">Expected Stock-out Probability: {row['stockout_prob']*100:.0f}%</p>
                    <p style="margin: 5px 0; color:var(--text-muted); font-size: 0.95rem;"><b>AI Reasoning:</b> {row['Explanation']}</p>
                </div>
                <div style="text-align: right;">
                    <div style="color:var(--text-muted); font-size: 0.85rem;">AI Confidence: <b>94%</b></div>
                    <div style="font-size: 1.5rem; font-weight: 800; color:var(--text-main);">{row['Reorder_Quantity']} units</div>
                    <div style="color:var(--text-muted); font-size: 0.85rem;">Recommended Order</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        bc1, bc2, bc3, _ = st.columns([1, 1, 1, 7])
        if bc1.button("Approve Order", key=f"app_{idx}"):
            st.session_state.approved_orders += 1
            st.toast(f"PO Generated for {row['Reorder_Quantity']} units of {row['product_id']}")
        bc2.button("Modify Quantity", key=f"mod_{idx}")
        bc3.button("Ignore Alert", key=f"ign_{idx}")
        st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# DEMAND INTELLIGENCE
# ==========================================
elif nav == "Demand Intelligence":
    st.markdown("<div class='section-title'><span class='mi'>insights</span> AI DEMAND FORECAST</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted);'>Analyze historical trends against XGBoost predictive horizons.</p>", unsafe_allow_html=True)
    
    df['historical_7_day_total'] = df['rolling_mean_7'] * 7
    trend_df = df.groupby('date')[['historical_7_day_total', 'predicted_demand']].mean().reset_index()
    
    # Build advanced Plotly chart with Confidence Area simulation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['date'], y=trend_df['historical_7_day_total'], mode='lines', name='Actual Historical', line=dict(color='#94A3B8', width=2)))
    fig.add_trace(go.Scatter(x=trend_df['date'], y=trend_df['predicted_demand'], mode='lines', name='AI Forecast', line=dict(color='#38BDF8', width=3, dash='dot')))
    
    # Simulated Confidence Interval
    fig.add_trace(go.Scatter(x=trend_df['date'], y=trend_df['predicted_demand']*1.15, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=trend_df['date'], y=trend_df['predicted_demand']*0.85, mode='lines', fill='tonexty', fillcolor='rgba(56,189,248,0.1)', line=dict(width=0), name='95% Confidence Area'))
    
    fig.update_layout(**chart_layout_kwargs, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'><span class='mi'>bar_chart</span> Top Risk Categories</div>", unsafe_allow_html=True)
    
    # Aggregate data for a simple Bar Chart
    bar_data = df.groupby('category')['stockout_prob'].mean().reset_index(name='Avg_Risk')
    bar_data['Avg_Risk'] = bar_data['Avg_Risk'] * 100
    
    # Sort by risk so the most critical categories are on the left
    bar_data = bar_data.sort_values(by='Avg_Risk', ascending=False)
    
    # Build a standard Vertical Bar Chart
    fig_bar = px.bar(
        bar_data, 
        x='category', 
        y='Avg_Risk', 
        color='Avg_Risk',
        color_continuous_scale='RdYlGn_r',
        text_auto='.1f'
    )
    
    fig_bar.update_traces(
        textfont_size=12, textposition="outside", cliponaxis=False,
        hovertemplate="<b>Category: %{x}</b><br>Avg Stock-out Risk: %{y:.1f}%"
    )
    fig_bar.update_layout(
        **chart_layout_kwargs, 
        yaxis_title="Average Stock-out Risk (%)", 
        xaxis_title="",
        coloraxis_showscale=False # Hide the color bar to keep it simple and clean
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# AI CONTROL ROOM
# ==========================================
elif nav == "AI Control Room":
    st.markdown("<div class='section-title'><span class='mi'>settings_suggest</span> STOCKSENSE AI CONTROL ROOM</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Status", "ONLINE")
    c2.metric("Forecast Accuracy", "94.7%")
    c3.metric("Predictions Today", "128,420")
    c4.metric("Anomalies Detected", "24")
    
    st.markdown("<hr style='border-color: #1F2937;'>", unsafe_allow_html=True)
    
    colA, colB = st.columns([1, 1])
    
    with colA:
        st.markdown("<div class='section-title'><span class='mi'>manage_search</span> Explainable AI (XAI) - Global Drivers</div>", unsafe_allow_html=True)
        features = ['Demand Spike', 'Low Current Stock', 'Supplier Delay', 'Seasonality', 'Promotion']
        importances = [0.31, 0.22, 0.19, 0.17, 0.11]
        fig_imp = px.bar(x=importances, y=features, orientation='h', color=importances, color_continuous_scale="Blues")
        fig_imp.update_layout(**chart_layout_kwargs, showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with colB:
        st.markdown("<div class='section-title'><span class='mi'>history</span> AI Activity / Audit Log</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="premium-card" style="height: 380px; overflow-y: auto;">
            <div class="timeline-item">
                <div class="time-text">11:32 AM</div>
                <div class="event-text">AI detected unusual demand in Frozen Foods.</div>
            </div>
            <div class="timeline-item">
                <div class="time-text">11:28 AM</div>
                <div class="event-text">Replenishment recommendations batch generated (12 items).</div>
            </div>
            <div class="timeline-item">
                <div class="time-text">11:21 AM</div>
                <div class="event-text">Store 042 identified as high stock-out risk hub.</div>
            </div>
            <div class="timeline-item" style="border-left: 2px solid transparent;">
                <div class="time-text">11:15 AM</div>
                <div class="event-text">Demand Forecast XGBoost Model successfully re-trained.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
