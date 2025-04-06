import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils import load_all_data
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("💤 Recovery Status Analysis")

# Load data
_, recovery_data_raw, _, _ = load_all_data()

# Data preprocessing
recovery_data = recovery_data_raw.pivot_table(
    index='sessionDate',
    columns='metric',
    values='value',
    aggfunc='first'
).reset_index()

# Convert sessionDate to datetime
recovery_data['sessionDate'] = pd.to_datetime(recovery_data['sessionDate'], dayfirst=True)

# Calculate rolling averages and trends
window_size = 7  # 7-day rolling window
for col in recovery_data.columns:
    if col != 'sessionDate':
        recovery_data[f'{col}_rolling_avg'] = recovery_data[col].rolling(window=window_size, min_periods=1).mean()
        recovery_data[f'{col}_trend'] = recovery_data[col].diff()

# Sidebar filters
st.sidebar.header("Analysis Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(recovery_data['sessionDate'].min(), recovery_data['sessionDate'].max())
)

# Filter data based on date selection
filtered_data = recovery_data[
    (recovery_data['sessionDate'].dt.date >= date_range[0]) &
    (recovery_data['sessionDate'].dt.date <= date_range[1])
]

# Main metrics dashboard
st.subheader("📊 Recovery Metrics Overview")
col1, col2, col3, col4 = st.columns(4)

# Calculate key metrics
if 'emboss_baseline_score' in filtered_data.columns:
    with col1:
        current_score = filtered_data['emboss_baseline_score'].iloc[-1]
        previous_score = filtered_data['emboss_baseline_score'].iloc[-2] if len(filtered_data) > 1 else current_score
        score_change = current_score - previous_score
        st.metric(
            "Current Recovery Score",
            f"{current_score:.1f}",
            f"{score_change:+.1f}",
            delta_color="normal" if score_change >= 0 else "inverse"
        )
    
    with col2:
        avg_score = filtered_data['emboss_baseline_score'].mean()
        st.metric("Average Recovery Score", f"{avg_score:.1f}")
    
    with col3:
        low_recovery_days = len(filtered_data[filtered_data['emboss_baseline_score'] < 50])
        st.metric("Low Recovery Days", f"{low_recovery_days}")
    
    with col4:
        recovery_trend = filtered_data['emboss_baseline_score_trend'].mean()
        st.metric(
            "Recovery Trend",
            "Improving" if recovery_trend > 0 else "Declining",
            f"{recovery_trend:+.2f} per day"
        )

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Recovery Trends", "Component Analysis", "Recovery Insights"])

with tab1:
    # Recovery trends visualization
    if 'emboss_baseline_score' in filtered_data.columns:
        fig_trends = go.Figure()
        
        # Add actual scores
        fig_trends.add_trace(go.Scatter(
            x=filtered_data['sessionDate'],
            y=filtered_data['emboss_baseline_score'],
            mode='lines+markers',
            name='Daily Score',
            line=dict(color='blue')
        ))
        
        # Add rolling average
        fig_trends.add_trace(go.Scatter(
            x=filtered_data['sessionDate'],
            y=filtered_data['emboss_baseline_score_rolling_avg'],
            mode='lines',
            name='7-day Rolling Average',
            line=dict(color='red', dash='dash')
        ))
        
        # Add reference lines
        fig_trends.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Optimal Recovery")
        fig_trends.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Recovery Threshold")
        
        fig_trends.update_layout(
            title="Recovery Score Trends",
            xaxis_title="Date",
            yaxis_title="Recovery Score",
            height=500
        )
        st.plotly_chart(fig_trends, use_container_width=True)

with tab2:
    # Component analysis
    st.subheader("Recovery Components Analysis")
    
    # Get all component metrics (excluding emboss_baseline_score and date)
    component_metrics = [col for col in filtered_data.columns 
                        if col not in ['sessionDate', 'emboss_baseline_score'] 
                        and not col.endswith(('_rolling_avg', '_trend'))]
    
    if component_metrics:
        # Create correlation heatmap
        corr_matrix = filtered_data[component_metrics].corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Recovery Components Correlation",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Show component trends
        for metric in component_metrics[:3]:  # Show first 3 components
            fig_component = go.Figure()
            fig_component.add_trace(go.Scatter(
                x=filtered_data['sessionDate'],
                y=filtered_data[metric],
                mode='lines+markers',
                name=metric.replace('_baseline_composite', '').title()
            ))
            fig_component.update_layout(
                title=f"{metric.replace('_baseline_composite', '').title()} Trend",
                xaxis_title="Date",
                yaxis_title="Score"
            )
            st.plotly_chart(fig_component, use_container_width=True)

with tab3:
    # Recovery insights and recommendations
    st.subheader("📈 Recovery Insights")
    
    # Calculate recovery patterns
    if 'emboss_baseline_score' in filtered_data.columns:
        # Identify recovery patterns
        low_recovery_periods = []
        current_period = []
        
        for idx, row in filtered_data.iterrows():
            if row['emboss_baseline_score'] < 50:
                current_period.append(row['sessionDate'])
            elif current_period:
                if len(current_period) >= 3:  # Only consider periods of 3+ days
                    low_recovery_periods.append((current_period[0], current_period[-1]))
                current_period = []
        
        # Display insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recovery Patterns**")
            if low_recovery_periods:
                st.warning("⚠️ Extended low recovery periods detected:")
                for start, end in low_recovery_periods:
                    st.write(f"- {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
            else:
                st.success("✅ No extended low recovery periods detected")
        
        with col2:
            st.write("**Recovery Recommendations**")
            current_score = filtered_data['emboss_baseline_score'].iloc[-1]
            
            if current_score < 50:
                st.error("🔴 Immediate Action Required")
                st.write("""
                - Reduce training intensity by 50%
                - Focus on active recovery
                - Increase sleep duration
                - Consider complete rest day
                """)
            elif current_score < 70:
                st.warning("🟡 Caution Needed")
                st.write("""
                - Reduce training intensity by 25%
                - Prioritize recovery protocols
                - Monitor sleep quality
                - Include active recovery sessions
                """)
            else:
                st.success("🟢 Optimal Recovery")
                st.write("""
                - Maintain current training load
                - Continue recovery protocols
                - Monitor for early signs of fatigue
                - Maintain good sleep habits
                """)

# Additional recommendations based on trends
st.subheader("📋 Action Items")
if 'emboss_baseline_score' in filtered_data.columns:
    recent_trend = filtered_data['emboss_baseline_score_trend'].iloc[-5:].mean()
    
    if recent_trend < -0.5:
        st.warning("""
        📉 **Declining Recovery Trend Detected**
        - Review recent training load
        - Consider implementing additional recovery protocols
        - Schedule recovery-focused sessions
        - Monitor sleep and nutrition
        """)
    elif recent_trend > 0.5:
        st.success("""
        📈 **Improving Recovery Trend Detected**
        - Current recovery strategies are effective
        - Consider gradually increasing training load
        - Maintain current recovery protocols
        - Continue monitoring recovery metrics
        """)
    else:
        st.info("""
        ➡️ **Stable Recovery Pattern**
        - Maintain current training and recovery balance
        - Continue regular monitoring
        - Focus on consistency in recovery protocols
        """)
