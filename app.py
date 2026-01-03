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
        
        # FIXED: Handle DD/MM/YYYY format properly
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', dayfirst=True)
        
        # Aggregate weekly total sales
        df['Week'] = df['Order Date'].dt.to_period('W').dt.to_timestamp()
        weekly = df.groupby('Week')['Sales'].sum().reset_index()
        
        st.write("Your weekly sales preview (last 10 weeks):")
        st.dataframe(weekly.tail(10))
        
        if st.button("Generate Forecast (next 8 weeks)"):
            with st.spinner("Training Prophet model..."):
                prophet_df = weekly.rename(columns={'Week': 'ds', 'Sales': 'y'})
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=8, freq='W')
                forecast = m.predict(future)
                
                st.success("Forecast ready!")
                st.write("Predicted sales for next 8 weeks:")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8))
                
                # Interactive chart
                fig = px.line(forecast, x='ds', y='yhat', 
                              title='Weekly Sales Forecast',
                              labels={'ds': 'Date', 'yhat': 'Predicted Sales ($)'})
                fig.add_scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical Sales')
                st.plotly_chart(fig)
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Tip: Make sure the CSV has 'Order Date' and 'Sales' columns. Try the format fix above.")