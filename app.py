import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import os

st.title("AIStockPredict MVP – Sales & Inventory Forecast")

st.markdown("""
Upload your sales CSV file.  
We forecast **future sales revenue or quantity** to help estimate inventory needs.  
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
        
        # === Automatically detect sales/quantity/revenue column ===
        sales_col = None
        is_quantity = False
        possible_sales_names = [
            'Sales', 'sales', 'Revenue', 'revenue', 'Weekly_Sales', 'weekly_sales',
            'SaleAmount', 'Amount', 'TotalSales', 'Profit',
            'Quantity', 'quantity', 'Units', 'units', 'Qty', 'qty'
        ]
        for col in possible_sales_names:
            if col in df.columns:
                sales_col = col
                # Check if it's likely quantity (for labeling)
                if 'quantity' in col.lower() or 'units' in col.lower() or 'qty' in col.lower():
                    is_quantity = True
                break
        
        if sales_col is None:
            raise ValueError(
                "No sales/revenue/quantity column found. Please ensure your CSV has one of these columns: " +
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
        
        label_unit = "Units" if is_quantity else "Sales ($)"
        
        st.subheader(f"Your Weekly {label_unit} Preview (last 10 weeks)")
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
                
                st.subheader(f"Predicted {label_unit} for Next 8 Weeks")
                st.dataframe(
                    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8)
                    .rename(columns={
                        'ds': 'Week Starting',
                        'yhat': f'Predicted {label_unit}',
                        'yhat_lower': 'Lower Bound',
                        'yhat_upper': 'Upper Bound'
                    })
                    .round(2 if is_quantity else 0)  # Round to integers for quantity, decimals for sales
                )
                
                # Interactive Plotly chart
                fig = px.line(
                    forecast,
                    x='ds',
                    y='yhat',
                    title=f'Weekly {label_unit} Forecast',
                    labels={'ds': 'Date', 'yhat': f'{label_unit}'}
                )
                fig.add_scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                )
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("""
        **Common fixes:**
        • Make sure your CSV has a date column and a sales/revenue/quantity column
        • Date format can be MM/DD/YYYY, DD/MM/YYYY or YYYY-MM-DD
        • Try downloading a clean dataset from Kaggle (Superstore or Walmart)
        • First row should be headers (not blank or merged cells)
        • If problem persists, share the first few column names from your CSV
        """)