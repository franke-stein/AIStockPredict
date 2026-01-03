import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
                # Prepare data for statsmodels (index must be datetime)
                weekly.set_index('Week', inplace=True)
                weekly.index = pd.to_datetime(weekly.index)
                
                # Simple Holt-Winters Exponential Smoothing (trend + seasonality)
                model = ExponentialSmoothing(
                    weekly['Sales'],
                    seasonal='add',
                    seasonal_periods=4
                ).fit()
                
                forecast = model.forecast(8)
                
                # Create forecast DataFrame for display
                forecast_dates = pd.date_range(
                    start=weekly.index[-1] + pd.Timedelta(weeks=1),
                    periods=8,
                    freq='W'
                )
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted Sales ($)': forecast.values
                })
                
                st.success("Forecast ready!")
                st.write("Predicted sales for next 8 weeks:")
                st.dataframe(forecast_df)
                
                # Interactive chart
                fig = px.line(
                    weekly.reset_index(),
                    x='Week',
                    y='Sales',
                    title='Weekly Sales + Forecast',
                    labels={'Sales': 'Actual Sales ($)', 'Week': 'Date'}
                )
                fig.add_scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Predicted Sales ($)'],
                    mode='lines',
                    name='Forecast',
                    line=dict(dash='dash', color='red')
                )
                st.plotly_chart(fig)
                
    except Exception as e:
        st.error(f"Error processing your file: {str(e)}")
        st.info("""
        Common fixes:
        • Make sure your CSV has columns 'Order Date' and 'Sales'
        • Date format should be DD/MM/YYYY (common in Superstore dataset)
        • Try downloading a fresh copy from Kaggle
        """)