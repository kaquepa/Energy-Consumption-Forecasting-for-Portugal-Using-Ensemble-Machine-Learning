import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))
from config import Config_preprocessor

Config = Config_preprocessor()
st.set_page_config(
    page_title="Energy Consumption Forecasting for Portugal",
    page_icon="📊",
    layout="wide",
)

@st.cache_data
def load_data(path):
    """Load and cache dataset."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)

DATA_PATH = Config.dataset_trainer_path 
df = load_data(DATA_PATH)

# Identify target column
if 'corrigido_temperatura' in df.columns:
    target = 'corrigido_temperatura'
else:
    targets = [c for c in df.columns if 'target' not in c.lower() and c != 'date']
    target = targets[0] if targets else df.columns[1]

st.sidebar.header("Controls")

# Date range selector
date_range = st.sidebar.date_input(
    "Date Range",
    value=[df.date.min(), df.date.max()],
    min_value=df.date.min(),
    max_value=df.date.max(),
    help="Filter data by date range"
)

# Filter data
df_filtered = df[
    (df.date >= pd.to_datetime(date_range[0])) &
    (df.date <= pd.to_datetime(date_range[1]))
].copy()

# Metrics in sidebar
st.sidebar.metric("Records", f"{len(df_filtered):,}")
st.sidebar.divider()

st.title("Energy Consumption Forecasting for Portugal")
st.caption(
    f"Comprehensive exploratory analysis of **{len(df_filtered):,}** days of electricity demand data "
    f"(**{df_filtered.date.min().date()}** to **{df_filtered.date.max().date()}**)"
)
st.divider()
mean = df_filtered[target].mean()
std = df_filtered[target].std()
cv = (std / mean) * 100
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric(
        label="Average Demand",
        value=f"{mean:.1f} GWh",
        help="Mean daily consumption"
    )

with k2:
    st.metric(
        label="Volatility",
        value=f"{std:.1f} GWh",
        help="Standard deviation of consumption"
    )
with k3:
    st.metric(
        label="Minimum",
        value=f"{df_filtered[target].min():.1f} GWh",
        delta=f"{((df_filtered[target].min()/mean - 1)*100):.1f}%"
    )

with k4:
    st.metric(
        label="Maximum",
        value=f"{df_filtered[target].max():.1f} GWh",
        delta=f"{((df_filtered[target].max()/mean - 1)*100):.1f}%"
    )

with k5:
    st.metric(
        label="Coeff. of Variation",
        value=f"{cv:.1f}%",
        help="Relative variability measure (std / mean)* 100)"
    )

st.divider()
tabs = st.tabs([
    "Demand Dynamics",
    "Seasonality & Cycles",
    "Variability & Risk",
    "Weather Sensitivity"
])


with tabs[0]:
    st.subheader("Long Term Demand Evolution")

    # Calculate moving averages
    df_filtered["ma_30"] = df_filtered[target].rolling(30, center=True).mean()
    df_filtered["ma_365"] = df_filtered[target].rolling(365, center=True).mean()

    # Create figure
    fig = go.Figure()
    # Daily demand
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered[target],
        name="Daily demand",
        line=dict(color='rgba(99, 110, 250, 0.5)', width=1),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Consumption: %{y:.2f} GWh<extra></extra>'
    ))
    
    # 30-day MA
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['ma_30'],
        name="30-day trend",
        line=dict(color='rgba(239, 85, 59, 0.8)', width=2, dash='dot'),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>30-day MA: %{y:.2f} GWh<extra></extra>'
    ))
    
    # 365-day MA
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['ma_365'],
        name="Long-term trend",
        line=dict(color='rgba(0, 204, 150, 1)', width=3),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>365-day MA: %{y:.2f} GWh<extra></extra>'
    ))

    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Consumption (GWh)",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Seasonal Structure")

    # Create temporal features
    df_filtered["year"] = df_filtered.date.dt.year
    df_filtered["month"] = df_filtered.date.dt.month

    # Pivot table for heatmap
    pivot = df_filtered.pivot_table(
        index="year",
        columns="month",
        values=target,
        aggfunc="mean"
    )
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Heatmap
    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Year", color="Consumption (GWh)"),
        x=month_names,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        text_auto='.1f'
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekly Consumption Patterns")
    df_filtered['day_of_week'] = df_filtered.date.dt.dayofweek
    df_filtered['is_weekend'] = df_filtered['day_of_week'].isin([5, 6])   
    weekly_data = df_filtered.groupby('day_of_week')[target].agg(['mean', 'std'])
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig = go.Figure()
    fig.add_trace(go.Bar(
            x=day_names,
            y=weekly_data['mean'].values,
            error_y=dict(type='data', array=weekly_data['std'].values, visible=True),
            marker_color=['#3498db']*5 + ['#e74c3c']*2,
            text=[f'{v:.1f}' for v in weekly_data['mean'].values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Average: %{y:.2f} GWh<br>Std Dev: %{error_y.array:.2f}<extra></extra>'
        ))
        
    fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Average Consumption (GWh)",
            height=450,
            showlegend=False,
            template='plotly_white'
        )
    st.plotly_chart(fig, use_container_width=True)
        
    # Weekend vs Weekday comparison
    weekend_avg = df_filtered[df_filtered['is_weekend']][target].mean()
    weekday_avg = df_filtered[~df_filtered['is_weekend']][target].mean()
    reduction = ((weekday_avg - weekend_avg) / weekday_avg) * 100 
    c1, c2, c3 = st.columns(3)
    c1.metric("Weekday Average", f"{weekday_avg:.2f} GWh")
    c2.metric("Weekend Average", f"{weekend_avg:.2f} GWh")
    c3.metric("Weekend Reduction", f"{reduction:.1f}%", delta_color="inverse")

with tabs[2]:
    st.subheader("Distribution & Extreme Behavior")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Histogram Distribution")
        
        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_filtered[target],
            nbinsx=80,
            name='Distribution',
            marker_color='rgba(99, 110, 250, 0.7)',
            hovertemplate='Range: %{x:.1f} GWh<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean and median lines
        fig.add_vline(
            x=mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean:.1f}",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=df_filtered[target].median(),
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: {df_filtered[target].median():.1f}",
            annotation_position="top left"
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Consumption (GWh)",
            yaxis_title="Frequency",
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Distribution highlighting skewness, tail risk and extreme consumption events")
    
    with col2:
        st.markdown("#####  Box Plot Analysis")
        # Box plot
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df_filtered[target],
            name='Consumption',
            marker_color='lightblue',
            boxmean='sd',
            hovertemplate='<b>Consumption</b><br>Value: %{y:.2f} GWh<extra></extra>'
        ))
        
        fig.update_layout(
            height=400,
            yaxis_title="Consumption (GWh)",
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Box plot revealing quartiles, median, and outlier detection")
    
    # Outlier detection 
    st.subheader("Outlier Detection (IQR Method)")
        
    # IQR calculation
    Q1 = df_filtered[target].quantile(0.25)
    Q3 = df_filtered[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
        
    outliers_low = df_filtered[df_filtered[target] < lower_bound]
    outliers_high = df_filtered[df_filtered[target] > upper_bound]
    total_outliers = len(outliers_low) + len(outliers_high)
        
     # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lower Bound", f"{lower_bound:.1f} GWh")
    c2.metric("Upper Bound", f"{upper_bound:.1f} GWh")
    c3.metric("Low Outliers", len(outliers_low), f"{len(outliers_low)/len(df_filtered)*100:.2f}%")
    c4.metric("High Outliers", len(outliers_high), f"{len(outliers_high)/len(df_filtered)*100:.2f}%")
        
    # Visualization
    fig = go.Figure()
    # Normal data
    fig.add_trace(go.Scatter(
            x=df_filtered['date'],
            y=df_filtered[target],
            mode='lines',
            name='Consumption',
            line=dict(color='steelblue', width=1),
            opacity=0.6,
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f} GWh<extra></extra>'
    ))
        
    # Bounds
    fig.add_hline(
            y=lower_bound,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Lower: {lower_bound:.0f} GWh",
            annotation_position="left"
    )
    fig.add_hline(
            y=upper_bound,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Upper: {upper_bound:.0f} GWh",
            annotation_position="left"
    )
        
    # Low outliers
    if len(outliers_low) > 0:
        fig.add_trace(go.Scatter(
                x=outliers_low['date'],
                y=outliers_low[target],
                mode='markers',
                name='Low Outliers',
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f} GWh<br><b>Low Outlier</b><extra></extra>'
        ))
        
    # High outliers
    if len(outliers_high) > 0:
        fig.add_trace(go.Scatter(
                x=outliers_high['date'],
                y=outliers_high[target],
                mode='markers',
                name='High Outliers',
                marker=dict(color='orange', size=8, symbol='x'),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f} GWh<br><b>High Outlier</b><extra></extra>'
        ))
        
    fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Consumption (GWh)",
            hovermode='closest',
            template='plotly_white'
    )
        
    st.plotly_chart(fig, use_container_width=True)
   
with tabs[3]:
    st.subheader("Weather Demand Sensitivity")

    # Find weather columns
    weather_cols = [c for c in df_filtered.columns if any(
        w in c.lower() for w in ["temp", "wind", "humidity", "rain", "solar", "precipitation", "radiation"]
    )]
    
    if len(weather_cols) == 0:
        st.warning("No weather columns found in dataset")
    else:
        weather_feature = st.selectbox(
            "🌤️ Select Weather Variable",
            weather_cols,
            key='weather_select',
            help="Choose a weather variable to analyze its relationship with consumption"
        )

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter plot with trendline
            fig = px.scatter(
                df_filtered,
                x=weather_feature,
                y=target,
                trendline="ols",
                opacity=0.35,
                labels={
                    weather_feature: weather_feature.replace('_', ' ').title(),
                    target: 'Consumption (GWh)'
                },
                color_discrete_sequence=['steelblue']
            )
            
            fig.update_layout(
                height=500,
                template='plotly_white'
            )
            fig.update_traces(
                hovertemplate='<b>%{x:.2f}</b><br>Consumption: %{y:.2f} GWh<extra></extra>'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation analysis
            r = df_filtered[[weather_feature, target]].corr().iloc[0, 1]
            
            st.metric("Pearson Correlation", f"{r:.4f}")
            # Interpret strength
            if abs(r) > 0.7:
                strength = "Very Strong"
                color = "#e74c3c"
            elif abs(r) > 0.5:
                strength = "Strong"
                color = "#e67e22"
            elif abs(r) > 0.3:
                strength = "Moderate"
                color = "#f39c12"
            else:
                strength = "Weak"
                color = "#95a5a6"
            
            st.markdown(
                f"**Strength:** <span style='color:{color}; font-weight:bold; font-size:1.1em'>{strength}</span>",
                unsafe_allow_html=True
            )
            
            st.markdown(f"""
            **Interpretation:**  
            {'**Negative correlation** - As this variable increases, consumption tends to decrease' 
             if r < 0 else 
             '**Positive correlation** - As this variable increases, consumption tends to increase'}
            """)


       
st.divider()

st.markdown("""
<div style="
    text-align: center;
    color: #9CA3AF;
    font-size: 0.75rem;
    padding: 1.5rem 0;
">
    <b>Energy Intelligence Platform</b> | Comprehensive Exploratory Data Analysis<br>
</div>
""", unsafe_allow_html=True)