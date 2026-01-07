
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

import sys

# Add code/ to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "code"))

from config import Config_dashboard
from utils.api_client import APIClient, get_cached_forecast
from utils.formatters import *
from utils.formatters import Formatters
from utils.chart_builder import ChartForecast



Config = Config_dashboard()
chart = ChartForecast()
formatter = Formatters()

st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%); }
    .stApp { max-width: 1400px; margin: 0 auto; }
    
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
        overflow-wrap: break-word;  
        max-width: 100%;  
    }
    
    .sub-header {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #6c757d !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #2d3748 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }
    
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }
    
    .weather-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: white;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .weather-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .weather-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 0.75rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .weather-card .emoji { font-size: 2.5rem; margin: 0.3rem 0; }
    .weather-card .condition { font-size: 1rem; font-weight: 600; margin: 0.3rem 0; }
    .weather-card .detail { font-size: 0.8rem; opacity: 0.95; margin: 0.2rem 0; }
    
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    .agreement-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    .agreement-card p { margin: 0.3rem 0; font-size: 0.85rem; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #e0e0e0, transparent);
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def convert_forecast_format(forecast_data):
    """Convert predictor format to dashboard format."""
    if not forecast_data:
        return None
    predictions_dict = forecast_data.get('predictions', {})
    predictions_list = []
    # Sort by day number
    sorted_keys = sorted([k for k in predictions_dict.keys() if k.startswith('day_')],
                        key=lambda x: int(x.replace('day_', '')))
    
    for key in sorted_keys:
        pred = predictions_dict[key]
        day_num = int(key.replace('day_', ''))
        prediction_value = pred.get('predicted')

        if prediction_value is None or not isinstance(prediction_value, (int, float)):
            continue
        converted = {
            "day": day_num,
            "date": pred.get('date', ''),
            "prediction": float(prediction_value),  
            "confidence": {"level": "high"},
            "weather": pred.get('weather', {}),
            "metrics": pred.get('metrics', {})
        }
        
        predictions_list.append(converted)
    
    result = {
        "predictions": predictions_list,
        "last_known": forecast_data.get('last_known', {}),
        "generated_at": forecast_data.get('generated_at')
    }

    return result
def load_historical_data(days):
    """Load historical data with fallback."""
    possible_paths = [
        Path(__file__).parent.parent / "data/preprocessor/dataset_production_final.csv"
    ]
    
    for dataset_path in possible_paths:
        if dataset_path.exists():
            try:
                df = pd.read_csv(dataset_path)
                
                date_col = None
                for col in df.columns:
                    if col.lower() in ["date", "data", "timestamp"]:
                        date_col = col
                        break
                
                if not date_col:
                    continue
                
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                target_col = None
                for col in df.columns:
                    if 'corrigido_temperatura' in col.lower() or 'target' in col.lower():
                        target_col = col
                        break
                
                if not target_col:
                    continue
                
                df = df[[date_col, target_col]].rename(columns={
                    date_col: "date",
                    target_col: "target_next_day"
                })
                
                df = df.sort_values("date").tail(days).reset_index(drop=True)
                return df
            
            except Exception:
                continue
    
    return pd.DataFrame({
        "date": pd.date_range(end=datetime.today(), periods=days),
        "target_next_day": [None] * days
    })

def render_kpi_cards(next_day_pred, forecast_data):
    """Render KPI cards."""
    col1, col2, col3, col4 = st.columns(4)
    # CARD 1: Today's Forecast
    with col1:
        if next_day_pred:
            pred_value = next_day_pred.get("predicted")
            
            if isinstance(pred_value, (int, float)):
                last_known = forecast_data.get('last_known', {})
                last_known_value = last_known.get('consumption')
                
                if isinstance(last_known_value, (int, float)) and last_known_value != 0:
                    change_pct = (pred_value - last_known_value) / last_known_value * 100
                else:
                    change_pct = 0
                    
                st.metric(
                    label="Today's Forecast",
                    value=formatter.format_gwh(pred_value),
                    delta=f"{change_pct:+.2f}%"
                )
            else:
                st.metric(label="Today's Forecast", value="N/A", delta="0%")
        else:
            st.metric(label="Today's Forecast", value="N/A", delta="0%")

    with col2:
        if forecast_data and 'predictions' in forecast_data and len(forecast_data['predictions']) > 0:
            day1 = forecast_data['predictions'][0]
            metrics = day1.get('metrics', {})
            mape = metrics.get('MAPE')
            st.metric(
                label="Model Accuracy",
                value=f"{mape:.2f}%",
                delta="MAPE",
                delta_color="inverse"
            )
        else:
            st.metric(label="Model Accuracy", value="N/A")
    
    # CARD 3: Forecast Horizon
    with col3:
        if forecast_data and 'predictions' in forecast_data:
            num_days = len(forecast_data['predictions'])
            st.metric(
                label="Forecast Horizon",
                value=f"{num_days} days",
                delta="Multi-horizon"
            )
        else:
            st.metric(label="Forecast Horizon", value="N/A")
    
    # CARD 4: Last Update
    with col4:
        if forecast_data and 'generated_at' in forecast_data:
            generated_at = forecast_data.get('generated_at')
            try:
                dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                time_str = dt.strftime("%H:%M")
                st.metric(
                    label="Last Update",
                    value=time_str,
                    delta="Live"
                )
            except:
                st.metric(label="Last Update", value="N/A")
        else:
            st.metric(label="Last Update", value="N/A")
def render_forecast_table(forecast_data):
    """Render forecast table with detailed change calculations."""
    st.markdown('<div class="section-header"> Detailed Forecast</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
    
    if not forecast_data or 'predictions' not in forecast_data:
        st.error("No forecast data available")
        return
    
    # Get last known real value
    last_known_value = None
    last_known_date = None
    if 'last_known' in forecast_data:
        last_known_value = forecast_data['last_known'].get('consumption')
        last_known_date = forecast_data['last_known'].get('date')
    
   
    table_data = []
    
    for i, pred in enumerate(forecast_data['predictions']):
        day = pred['day']
        date_str = pred['date']
        prediction = pred['prediction']
        confidence = pred.get('confidence', {})
        
        conf_level = confidence.get('level', 'medium')
        conf_emoji = formatter.get_confidence_emoji(conf_level)
        
        # Calculate change with explicit logic
        if i == 0:
            # Day +1: Compare with last real value
            if last_known_value is not None and last_known_value > 0:
                change_value = ((prediction - last_known_value) / last_known_value) * 100
                change = f"{change_value:+.2f}%"
            else:
                change = "-"
        else:
            # Day +2 onwards: Compare with previous prediction
            prev_pred = forecast_data['predictions'][i-1]['prediction']
            if prev_pred > 0:
                change_value = ((prediction - prev_pred) / prev_pred) * 100
                change = f"{change_value:+.2f}%"
            else:
                change = "-"
        
        table_data.append({
            'Day': f"Day +{day}",
            'Date': formatter.format_date(date_str),
            'Forecast': formatter.format_gwh(prediction),
            'Confidence': f"{conf_emoji} {conf_level.upper()}",
            'Change': change
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_weather_cards(forecast_data):
    """Render weather as horizontal cards."""
    st.markdown('<div class="section-header">Weather Forecast (7 days)</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
    
    if not forecast_data or 'predictions' not in forecast_data:
        st.error("No weather data available")
        return
    
    predictions = forecast_data['predictions'][:7]
    
    if not predictions:
        st.warning("No predictions available")
        return
    
    cols = st.columns(7)
    for i, pred in enumerate(predictions):
        with cols[i]:
            weather = pred.get('weather', {})
            
            temp = weather.get('temperatura', 0) if weather else 0
            rain = weather.get('chuva', 0) if weather else 0
            if rain > 5:
                emoji = "🌧️"
                condition = "Rain"
            elif rain > 0:
                emoji = "⛅"
                condition = "Clouds"
            else:
                emoji = "☀️"
                condition = "Clear"
            
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.65rem; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    DAY +{pred['day']}
                </div>
                <div style="font-size: 0.8rem; color: #333; margin: 0.3rem 0; font-weight: 600;">
                    {formatter.format_date(pred['date'], '%a, %b %d')}
                </div>
                <div style="font-size: 2.5rem; margin: 0.5rem 0;">
                    {emoji}
                </div>
                <div style="font-size: 1.3rem; color: #2d3748; font-weight: 700; margin: 0.3rem 0;">
                    {temp}°C
                </div>
                <div style="font-size: 0.7rem; color: #666; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #f0f0f0;">
                    {condition} • {rain}mm
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div style="
            text-align: center;
            color: #9CA3AF;
            font-size: 0.7rem;
            padding: 0.75rem 0;
        ">
            Energy Intelligence Platform v2.1 <br>
            © 2025 · dkgraciano92@ua.pt
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main dashboard function."""
    st.markdown(f'<div class="main-header">{Config.PAGE_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{datetime.now().strftime("%A, %B %d, %Y")} • Live Dashboard</div>', unsafe_allow_html=True)
    
    api_client = APIClient(Config.API_BASE_URL)
    
    with st.spinner("Loading data..."):
        next_day_pred = api_client.get_next_day_prediction()
        forecast_data_raw = get_cached_forecast(Config.API_BASE_URL, Config.FORECAST_DAYS)
        forecast_data = convert_forecast_format(forecast_data_raw)
        historical_data = load_historical_data(Config.HISTORICAL_DAYS)
    
    if not forecast_data:
        st.error("Could not connect to API")
        st.stop()
    
    # KPI Cards
    render_kpi_cards(next_day_pred, forecast_data)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Consumption Forecast Chart
    st.markdown('<div class="section-header">Consumption Forecast</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
    
    if forecast_data and 'predictions' in forecast_data:
        fig = chart.create_forecast_chart(historical_data, forecast_data['predictions'],  title=f"Consumption Forecast - {Config.HISTORICAL_DAYS} days History + {Config.FORECAST_DAYS} Days Ahead")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Forecast Table
    render_forecast_table(forecast_data)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Weather Forecast
    render_weather_cards(forecast_data)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Footer
    render_footer()
    
    # Auto-refresh
    if st.sidebar.checkbox("Enable auto-refresh", value=False):
        st.sidebar.info(f"Refreshing every {Config.AUTO_REFRESH_SECONDS}s")
        import time
        time.sleep(Config.AUTO_REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()