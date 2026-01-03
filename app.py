import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.title("AIStockPredict MVP – Sales & Inventory Forecast")

st.markdown("Upload your sales CSV. We forecast **future sales revenue** to help estimate inventory needs.")

uploaded_file = st.file_uploader("Upload sales CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Handle DD/MM/YYYY format properly (common in Superstore dataset)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', dayfirst=True)
        
        # Aggregate weekly total sales
        df['Week'] = df['Order Date'].dt.to_period('W').dt.to_timestamp()
        weekly = df.groupby('Week')['Sales'].sum().reset_index()
        
        st.write("Your weekly sales preview (last 10 weeks):")
        st.dataframe(weekly.tail(10))
        
       # ... (keep everything until the forecast block)

if st.button("Generate Forecast (next 8 weeks)"):
    with st.spinner("Training simple model..."):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Prepare data
        weekly.set_index('Week', inplace=True)
        weekly.index = pd.to_datetime(weekly.index)
        
        # Simple Holt-Winters (good for trends + seasonality)
        model = ExponentialSmoothing(weekly['Sales'], seasonal='add', seasonal_periods=4).fit()
        
        forecast = model.forecast(8)
        
        # Create forecast df for display/chart
        forecast_dates = pd.date_range(start=weekly.index[-1] + pd.Timedelta(weeks=1), periods=8, freq='W')
        forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast.values})
        
        st.success("Forecast ready!")
        st.write("Predicted sales for next 8 weeks:")
        st.dataframe(forecast_df)
        
        # Chart
        fig = px.line(weekly.reset_index(), x='Week', y='Sales', title='Weekly Sales + Forecast')
        fig.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(dash='dash'))
        st.plotly_chart(fig)
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Tip: Make sure the CSV has 'Order Date' (DD/MM/YYYY) and 'Sales' columns.")
