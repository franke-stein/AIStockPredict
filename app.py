import streamlit as st
import pandas as pd
import altair as alt

st.title("AIStockPredict MVP – Sales & Inventory Forecast")

st.markdown("Upload your sales CSV. We forecast **future sales revenue** to help estimate inventory needs.")

uploaded_file = st.file_uploader("Upload sales CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Handle common DD/MM/YYYY format from Superstore dataset
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', dayfirst=True)
        
        # Aggregate weekly total sales
        df['Week'] = df['Order Date'].dt.to_period('W').dt.to_timestamp()
        weekly = df.groupby('Week')['Sales'].sum().reset_index()
        
        st.write("Your weekly sales preview (last 10 weeks):")
        st.dataframe(weekly.tail(10))
        
        if st.button("Generate Forecast (next 8 weeks)"):
            with st.spinner("Training simple model..."):
                # Prepare data for statsmodels
                weekly.set_index('Week', inplace=True)
                weekly.index = pd.to_datetime(weekly.index)
                
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                
                # Holt-Winters Exponential Smoothing
                model = ExponentialSmoothing(
                    weekly['Sales'],
                    seasonal='add',
                    seasonal_periods=4
                ).fit()
                
                forecast = model.forecast(8)
                
                # Create forecast DataFrame
                forecast_dates = pd.date_range(
                    start=weekly.index[-1] + pd.Timedelta(weeks=1),
                    periods=8,
                    freq='W'
                )
                forecast_df = pd.DataFrame({
                    'Week': forecast_dates,
                    'Sales': forecast.round(2).values,
                    'Type': 'Forecast'
                })
                
                # Combine historical + forecast for chart
                historical_df = weekly.reset_index().assign(Type='Historical')
                chart_data = pd.concat([historical_df, forecast_df], ignore_index=True)
                
                # Create interactive Altair chart
                chart = alt.Chart(chart_data).mark_line().encode(
                    x='Week:T',
                    y='Sales:Q',
                    color='Type:N',
                    tooltip=['Week', 'Sales', 'Type']
                ).properties(
                    title='Weekly Sales & Forecast',
                    width=700,
                    height=400
                )
                
                st.success("Forecast ready!")
                st.write("**Predicted sales for next 8 weeks:**")
                st.dataframe(forecast_df[['Week', 'Sales']])
                
                st.altair_chart(chart, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Tips: Ensure CSV has 'Order Date' (DD/MM/YYYY) and 'Sales' columns. Try a fresh Kaggle download.")