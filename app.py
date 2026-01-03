import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import os

st.title("AIStockPredict MVP – Sales & Inventory Forecast")

st.markdown("""
Upload your sales CSV file.  
We forecast **future sales revenue** to help estimate inventory needs.  
""")

uploaded_file = st.file_uploader("Upload sales CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # === Automatically detect date column ===
        date_col = None
        possible_date_names = [
            'Order Date', 'Date', 'order_date', 'Order_Date', 'ds',
            'transaction_date', 'OrderDate', 'SaleDate', 'Ship Date'
        ]
        for col in possible_date_names:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(
                "No date column found. Please ensure your CSV has one of these columns: " +
                ", ".join(possible_date_names)
            )
        
        # === Automatically detect sales/revenue column ===
        sales_col = None
        possible_sales_names = [
            'Sales', 'sales', 'Revenue', 'revenue', 'Weekly_Sales', 'weekly_sales',
            'SaleAmount', 'Amount', 'TotalSales', 'Profit'
        ]
        for col in possible_sales_names:
            if col in df.columns:
                sales_col = col
                break
        
        if sales_col is None:
            raise ValueError(
                "No sales/revenue column found. Please ensure your CSV has one of these columns: " +
                ", ".join(possible_sales_names)
            )
        
        # Convert date column to datetime (try dayfirst for DD/MM/YYYY formats)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        
        # Drop rows with invalid dates
        df = df.dropna(subset=[date_col])
        
        if df.empty:
            raise ValueError("No valid date entries found after parsing.")
        
        # Aggregate to weekly totals
        df['Week'] = df[date_col].dt.to_period('W').dt.to_timestamp()
        weekly = df.groupby('Week')[sales_col].sum().reset_index()
        weekly = weekly.sort_values('Week')
        
        st.subheader("Your Weekly Sales Preview (last 10 weeks)")
        st.dataframe(weekly.tail(10))
        
        if st.button("Generate Forecast (next 8 weeks)"):
            with st.spinner("Training Prophet model..."):
                # Prepare data for Prophet
                prophet_df = weekly.rename(columns={'Week': 'ds', sales_col: 'y'})
                
                # Initialize and fit Prophet
                m = Prophet()
                m.fit(prophet_df)
                
                # Make future dataframe
                future = m.make_future_dataframe(periods=8, freq='W')
                forecast = m.predict(future)
                
                st.success("Forecast ready!")
                
                st.subheader("Predicted Sales for Next 8 Weeks")
                st.dataframe(
                    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8)
                    .rename(columns={
                        'ds': 'Week Starting',
                        'yhat': 'Predicted Sales',
                        'yhat_lower': 'Lower Bound',
                        'yhat_upper': 'Upper Bound'
                    })
                    .round(2)
                )
                
                # Interactive Plotly chart
                fig = px.line(
                    forecast,
                    x='ds',
                    y='yhat',
                    title='Weekly Sales Forecast',
                    labels={'ds': 'Date', 'yhat': 'Predicted Sales ($)'}
                )
                fig.add_scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    mode='lines',
                    name='Historical Sales',
                    line=dict(color='blue')
                )
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("""
        **Common fixes:**
        • Make sure your CSV has a date column and a sales/revenue column
        • Date format can be MM/DD/YYYY, DD/MM/YYYY or YYYY-MM-DD
        • Try downloading a clean dataset from Kaggle (Superstore or Walmart)
        • First row should be headers (not blank or merged cells)
        • If problem persists, share the first few column names from your CSV
        """)

