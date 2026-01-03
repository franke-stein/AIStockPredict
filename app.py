import streamlit as st
import pandas as pd
import numpy as np

st.title("AIStockPredict MVP – Simple Sales Forecast")

st.markdown("""
Upload your sales CSV.  
We forecast future sales revenue using a basic linear trend (no heavy libraries needed).
""")

uploaded_file = st.file_uploader("Upload sales CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Handle DD/MM/YYYY format common in Superstore dataset
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', dayfirst=True)
        
        # Aggregate weekly total sales
        df['Week'] = df['Order Date'].dt.to_period('W').dt.to_timestamp()
        weekly = df.groupby('Week')['Sales'].sum().reset_index()
        weekly = weekly.sort_values('Week')
        
        st.write("Your weekly sales preview (last 10 weeks):")
        st.dataframe(weekly.tail(10))
        
        if st.button("Generate Forecast (next 8 weeks)"):
            with st.spinner("Calculating simple linear forecast..."):
                # Simple linear regression using numpy (no statsmodels needed)
                weekly['time'] = np.arange(len(weekly))
                slope, intercept = np.polyfit(weekly['time'], weekly['Sales'], 1)
                
                # Future time points
                future_times = np.arange(len(weekly), len(weekly) + 8)
                forecast_sales = slope * future_times + intercept
                
                # Create forecast dates
                last_date = weekly['Week'].iloc[-1]
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                              periods=8, freq='W')
                
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates.strftime('%Y-%m-%d'),
                    'Predicted Sales ($)': forecast_sales.round(2)
                })
                
                st.success("Forecast ready!")
                st.write("**Predicted sales for next 8 weeks (linear trend):**")
                st.dataframe(forecast_df)
                
                # Simple line chart with matplotlib (fallback, reliable)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(weekly['Week'], weekly['Sales'], label='Historical', marker='o')
                ax.plot(forecast_dates, forecast_sales, label='Forecast', linestyle='--', marker='x', color='red')
                ax.set_title('Weekly Sales & Linear Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("""
        Quick fixes:
        • CSV must have 'Order Date' (DD/MM/YYYY) and 'Sales' columns
        • Try downloading fresh Superstore CSV from Kaggle
        • Date parsing failed? Reply with the exact error for help
        """)