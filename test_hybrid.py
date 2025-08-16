#!/usr/bin/env python3
"""
Test script for the hybrid Finnhub + yfinance approach
"""

import os
from finnhub_analysis import FinnhubAnalyzer

def test_hybrid_approach():
    """Test the hybrid data approach"""
    print("🧪 Testing Hybrid Data Approach")
    print("="*50)
    
    # Load API key from environment
    api_key = os.getenv('API_KEY') or os.getenv('FINNHUB_API_KEY')
    if not api_key:
        print("❌ No API key found in environment variables")
        print("Please set API_KEY or FINNHUB_API_KEY in your .env file")
        return
    
    try:
        # Initialize analyzer
        analyzer = FinnhubAnalyzer(api_key)
        print("✅ Analyzer initialized successfully")
        
        # Test symbol that might have Finnhub restrictions
        test_symbols = ["BA", "AAPL", "MSFT"]
        
        for symbol in test_symbols:
            print(f"\n{'='*30}")
            print(f"Testing {symbol}")
            print(f"{'='*30}")
            
            # Check data source availability
            print(analyzer.get_data_source_info(symbol))
            
            # Try to get historical data (this will test the fallback)
            print(f"\n📈 Testing historical data for {symbol}...")
            df = analyzer.get_historical_data(symbol, days=7)
            
            if df is not None and not df.empty:
                print(f"✅ Successfully got {len(df)} days of data for {symbol}")
                print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")
                print(f"   Data source: {'yfinance' if 'yfinance' in str(df) else 'Finnhub'}")
            else:
                print(f"❌ Failed to get data for {symbol}")
        
        print(f"\n{'='*50}")
        print("✅ Hybrid approach test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_hybrid_approach()
