 
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

class ChartForecast():
    def __init__(self):
        pass
    def create_forecast_chart(self,
        historical_data: pd.DataFrame,
        forecast_data: List[Dict[str, Any]],
        title: str
    ) -> go.Figure:
        """Create main forecast chart with historical data and predictions."""
        fig = go.Figure()
        
        # Convert dates to string format for Plotly
        historical_data = historical_data.copy()
        historical_data['date'] = pd.to_datetime(historical_data['date']).dt.strftime('%Y-%m-%d')
        
        # Historical line -
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['target_next_day'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='rgba(26, 84, 144, 1.0)', width=2),
            marker=dict(size=6),
            hovertemplate='<b>%{x}</b><br>Consumption: %{y:.1f} GWh<extra></extra>'
        ))
        
        valid_forecast = []
        for f in forecast_data:
            pred = f.get('prediction')
            if pred is not None and isinstance(pred, (int, float)):
                valid_forecast.append(f)
            else:
                # Skip invalid predictions
                continue
        
        if not valid_forecast:
            # No valid forecasts - return chart with only historical data
            fig.update_layout(
                title=dict(text=title, font=dict(size=18, color='#1a5490')),
                xaxis_title="Date",
                yaxis_title="Consumption (GWh)",
                height=500,
                template='plotly_white',
            )
            return fig
        
        # Forecast line (dashed blue)
        forecast_dates = [f['date'] for f in valid_forecast]
        forecast_values = [float(f['prediction']) for f in valid_forecast]  
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='rgba(26, 84, 144, 0.8)', width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>Predicted: %{y:.1f} GWh<extra></extra>'
        ))
        
        # Confidence bands
        if len(valid_forecast) > 0:
            # Get RMSE for confidence band 
            rmse_values = []
            for f in valid_forecast:
                metrics = f.get('metrics', {})
                rmse = metrics.get('RMSE')  
                if rmse is None or not isinstance(rmse, (int, float)):
                    rmse = 5.5
                rmse_values.append(float(rmse))
            upper_values = [float(f['prediction']) + rmse for f, rmse in zip(valid_forecast, rmse_values)]
            lower_values = [float(f['prediction']) - rmse for f, rmse in zip(valid_forecast, rmse_values)]
            
            # Upper bound
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=upper_values,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=lower_values,
                fill='tonexty',
                fillcolor='rgba(26, 84, 144, 0.15)',
                mode='lines',
                line=dict(width=0),
                name='68% Confidence',
                hoverinfo='skip'
            ))
        
        # Add weekend shading
        for idx, row in historical_data.iterrows():
            try:
                date_str = row['date']
                date_obj = pd.to_datetime(date_str)
                
                if date_obj.dayofweek >= 5:  # Saturday=5, Sunday=6
                    fig.add_vrect(
                        x0=date_str,
                        x1=date_str,
                        fillcolor="rgba(220, 53, 69, 0.1)",
                        layer="below",
                        line_width=0,
                    )
            except:
                continue
        
        # Today marker
        today = datetime.now().strftime('%Y-%m-%d')
        fig.add_shape(
            type="line",
            x0=today, x1=today,
            y0=0, y1=1,
            yref="paper",
            line=dict(color='rgba(40, 167, 69, 0.8)', width=2, dash='dot')
        )
        fig.add_annotation(
            x=today,
            y=1,
            yref="paper",
            text="Today",
            showarrow=False,
            yshift=10,
            font=dict(color='rgba(40, 167, 69, 0.8)')
        )
        
        # Layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color='#1a5490')),
            xaxis_title="Date",
            yaxis_title="Consumption (GWh)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        return fig

