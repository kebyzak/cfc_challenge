import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import load_all_data
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("🏋️‍♂️ Physical Capability Analysis")

# Load data
_, _, physical_data, _ = load_all_data()

# Data preprocessing
physical_data['testDate'] = pd.to_datetime(physical_data['testDate'], dayfirst=True)

# Calculate rolling averages and trends only for numeric columns
window_size = 3  # 3-test rolling window
numeric_columns = physical_data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if col not in ['testDate']:  # Exclude date column
        physical_data[f'{col}_rolling_avg'] = physical_data.groupby('movement')[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        physical_data[f'{col}_trend'] = physical_data.groupby('movement')[col].transform(
            lambda x: x.diff()
        )

# Sidebar filters
st.sidebar.header("Analysis Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(physical_data['testDate'].min(), physical_data['testDate'].max())
)

selected_movements = st.sidebar.multiselect(
    "Select Movements",
    options=physical_data['movement'].unique(),
    default=physical_data['movement'].unique()[:3]
)

# Filter data based on selections
filtered_data = physical_data[
    (physical_data['testDate'].dt.date >= date_range[0]) &
    (physical_data['testDate'].dt.date <= date_range[1]) &
    (physical_data['movement'].isin(selected_movements))
]

# Main metrics dashboard
st.subheader("📊 Physical Capability Overview")
col1, col2, col3, col4 = st.columns(4)

# Calculate key metrics
with col1:
    current_benchmark = filtered_data.groupby('movement')['benchmarkPct'].last()
    avg_benchmark = current_benchmark.mean()
    st.metric("Average Benchmark %", f"{avg_benchmark:.1f}%")

with col2:
    improvement_rate = filtered_data.groupby('movement')['benchmarkPct_trend'].mean().mean()
    st.metric(
        "Overall Improvement Rate",
        f"{improvement_rate:+.1f}% per test",
        delta_color="normal" if improvement_rate > 0 else "inverse"
    )

with col3:
    best_movement = current_benchmark.idxmax()
    best_score = current_benchmark.max()
    st.metric("Best Performing Movement", best_movement, f"{best_score:.1f}%")

with col4:
    recent_tests = len(filtered_data['testDate'].unique())
    st.metric("Recent Tests", f"{recent_tests}")

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Performance Trends", "Movement Analysis", "Capability Insights"])

with tab1:
    # Performance trends visualization
    fig_trends = go.Figure()
    
    for movement in selected_movements:
        movement_data = filtered_data[filtered_data['movement'] == movement]
        
        # Add actual scores
        fig_trends.add_trace(go.Scatter(
            x=movement_data['testDate'],
            y=movement_data['benchmarkPct'],
            mode='lines+markers',
            name=f'{movement} (Actual)',
            line=dict(width=2)
        ))
        
        # Add rolling average
        fig_trends.add_trace(go.Scatter(
            x=movement_data['testDate'],
            y=movement_data['benchmarkPct_rolling_avg'],
            mode='lines',
            name=f'{movement} (Trend)',
            line=dict(dash='dash', width=1)
        ))
    
    # Add reference lines
    fig_trends.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Benchmark Target")
    fig_trends.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Performance Threshold")
    
    fig_trends.update_layout(
        title="Physical Capability Trends by Movement",
        xaxis_title="Test Date",
        yaxis_title="Benchmark Percentage (%)",
        height=500
    )
    st.plotly_chart(fig_trends, use_container_width=True)

with tab2:
    # Movement analysis
    st.subheader("Movement-Specific Analysis")
    
    # Create correlation heatmap
    movement_metrics = ['benchmarkPct', 'load', 'velocity', 'power']
    available_metrics = [col for col in movement_metrics if col in filtered_data.columns]
    
    if len(available_metrics) > 1:
        corr_matrix = filtered_data[available_metrics].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Performance Metrics Correlation",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Movement comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance distribution
        fig_dist = px.box(
            filtered_data,
            x='movement',
            y='benchmarkPct',
            title="Performance Distribution by Movement"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Load vs. Velocity relationship if both metrics are available
        if 'load' in filtered_data.columns and 'velocity' in filtered_data.columns:
            fig_scatter = px.scatter(
                filtered_data,
                x='load',
                y='velocity',
                color='movement',
                size='benchmarkPct',
                title="Load vs. Velocity Relationship",
                labels={'load': 'Load (kg)', 'velocity': 'Velocity (m/s)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Load and velocity data not available for this analysis.")

with tab3:
    # Capability insights
    st.subheader("📈 Performance Insights")
    
    # Calculate performance patterns
    performance_patterns = []
    for movement in selected_movements:
        movement_data = filtered_data[filtered_data['movement'] == movement]
        if 'benchmarkPct_trend' in movement_data.columns and len(movement_data) >= 3:
            recent_trend = movement_data['benchmarkPct_trend'].iloc[-3:].mean()
            
            if recent_trend > 2:
                performance_patterns.append((movement, "rapidly improving"))
            elif recent_trend > 0:
                performance_patterns.append((movement, "gradually improving"))
            elif recent_trend < -2:
                performance_patterns.append((movement, "rapidly declining"))
            elif recent_trend < 0:
                performance_patterns.append((movement, "gradually declining"))
            else:
                performance_patterns.append((movement, "stable"))
        else:
            performance_patterns.append((movement, "insufficient data"))
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Patterns**")
        for movement, pattern in performance_patterns:
            if "improving" in pattern:
                st.success(f"✅ {movement}: {pattern}")
            elif "declining" in pattern:
                st.warning(f"⚠️ {movement}: {pattern}")
            elif "stable" in pattern:
                st.info(f"ℹ️ {movement}: {pattern}")
            else:
                st.warning(f"⚠️ {movement}: {pattern}")
    
    with col2:
        st.write("**Movement-Specific Recommendations**")
        for movement, pattern in performance_patterns:
            if "rapidly improving" in pattern:
                st.write(f"**{movement}**: Consider increasing load or complexity")
            elif "gradually improving" in pattern:
                st.write(f"**{movement}**: Maintain current progression")
            elif "rapidly declining" in pattern:
                st.write(f"**{movement}**: Review technique and reduce load")
            elif "gradually declining" in pattern:
                st.write(f"**{movement}**: Focus on form and consider deload")
            elif "stable" in pattern:
                st.write(f"**{movement}**: Fine-tune technique and load")
            else:
                st.write(f"**{movement}**: Collect more data for analysis")

# Additional recommendations
st.subheader("📋 Action Items")
if len(selected_movements) > 0 and 'benchmarkPct_trend' in filtered_data.columns:
    # Calculate overall performance trend
    overall_trend = filtered_data.groupby('movement')['benchmarkPct_trend'].mean().mean()
    
    if overall_trend > 1:
        st.success("""
        📈 **Strong Overall Improvement Detected**
        - Consider progressive overload
        - Focus on maintaining technique
        - Monitor for signs of overtraining
        - Document successful training strategies
        """)
    elif overall_trend < -1:
        st.error("""
        📉 **Performance Decline Detected**
        - Review recent training load
        - Consider deload week
        - Focus on technique
        - Assess recovery protocols
        """)
    else:
        st.info("""
        ➡️ **Stable Performance Pattern**
        - Maintain current training approach
        - Focus on consistency
        - Monitor for subtle improvements
        - Consider technique refinement
        """)

# Performance projections
st.subheader("🎯 Performance Projections")
if len(filtered_data) > 1 and 'benchmarkPct' in filtered_data.columns:
    # Calculate projected performance
    projections = []
    for movement in selected_movements:
        movement_data = filtered_data[filtered_data['movement'] == movement]
        if len(movement_data) > 1:
            # Simple linear projection
            x = np.arange(len(movement_data))
            y = movement_data['benchmarkPct'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Project next 3 tests
            next_tests = p(len(x) + np.arange(1, 4))
            projections.append((movement, next_tests))
    
    if projections:
        fig_proj = go.Figure()
        
        for movement, proj in projections:
            # Add historical data
            movement_data = filtered_data[filtered_data['movement'] == movement]
            fig_proj.add_trace(go.Scatter(
                x=movement_data['testDate'],
                y=movement_data['benchmarkPct'],
                mode='lines+markers',
                name=f'{movement} (Historical)',
                line=dict(width=2)
            ))
            
            # Add projections
            last_date = movement_data['testDate'].iloc[-1]
            proj_dates = [last_date + timedelta(days=14*i) for i in range(1, 4)]
            fig_proj.add_trace(go.Scatter(
                x=proj_dates,
                y=proj,
                mode='lines',
                name=f'{movement} (Projected)',
                line=dict(dash='dot', width=1)
            ))
        
        fig_proj.update_layout(
            title="Performance Projections (Next 6 Weeks)",
            xaxis_title="Date",
            yaxis_title="Benchmark Percentage (%)",
            height=500
        )
        st.plotly_chart(fig_proj, use_container_width=True)
