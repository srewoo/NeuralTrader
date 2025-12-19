
import asyncio
import sys
import os
import yfinance as yf

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from ml.inference import MLService

async def test_ml_prediction():
    print("Testing ML Prediction Pipeline...")
    service = MLService()
    
    symbol = "RELIANCE.NS"
    print(f"Fetching data for {symbol}...")
    
    # Fetch data directly
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")
    
    if hist.empty:
        print("Failed to fetch data (Market closed or network issue). Using synthetic data.")
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start='2023-01-01', periods=300)
        close = np.random.normal(2500, 50, 300).cumsum() # Random walk
        hist = pd.DataFrame({'Close': close}, index=dates)
        
    print(f"Data shape: {hist.shape}")
    
    try:
        print("Running training and prediction...")
        result = await service.predict_next_price("RELIANCE", hist, lookback_days=30)
        
        print("\nPrediction Result:")
        print(f"Symbol: {result['symbol']}")
        print(f"Current Price: {result['current_price']}")
        print(f"Predicted Price (Next Day): {result['predicted_price']}")
        print(f"Change: {result['change_pct']}%")
        print(f"Confidence Interval: {result['confidence_interval']}")
        
        # Validation
        if result['predicted_price'] > 0 and len(result['confidence_interval']) == 2:
            print("\nSUCCESS: Prediction pipeline returned valid structure.")
        else:
            print("\nFAILURE: Invalid prediction result.")
            
    except Exception as e:
        print(f"\nFAILURE: Prediction raised exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ml_prediction())
