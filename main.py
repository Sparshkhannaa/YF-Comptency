import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
from mapping import FEATURE_MAP
from features import Feature


def get_integer(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer!")


def get_symbol(prompt):
    ticker = input(prompt).strip().upper()
    if not ticker.isalnum():
        print("Invalid symbol—use letters and numbers only.")
        return get_symbol(prompt)
    return ticker


def get_feature_request(prompt):
    while True:
        try:
            feature_num = int(input(prompt))
            if feature_num in FEATURE_MAP:
                return feature_num
            else:
                print(f"Please enter a valid feature number between 1 and {len(FEATURE_MAP)}")
        except ValueError:
            print("Please enter a valid integer!")


def get_data(ticker, days):
    data = yf.Ticker(ticker)
    return data.history(period=f"{days}mo")


def process_feature_request(symbol, feature_request):
    try:
        feature_name = FEATURE_MAP[feature_request]
        
        feature = Feature(feature_name, f"Data for {feature_name}", "DataFrame", feature_request)
        
        print(f"Processing {feature_name} for {symbol}...")
        data = feature.process_feature(symbol)
        
        if data is not None:
            feature.display_data()
            print(f"\n{feature.get_summary()}")
            return data
        else:
            print(f"Failed to process {feature_name} for {symbol}")
            return None
            
    except Exception as e:
        print(f"Error processing feature {feature_request}: {str(e)}")
        return None


def main():
    print("=== STOCK ANALYSIS TOOL -SPARSH KHANNA===")
    print()
    
    symbol = get_symbol("Enter stock symbol (e.g., HOOD): ")
    
    days = get_integer("Enter number of months back to fetch data for:(VALID VALUES ARE 1,3,6,12) ")
    
    print(f"\nFetching {days} months of basic data for {symbol}...")
    
    basic_data = get_data(symbol, days)
    if not basic_data.empty:
        print(f"Successfully fetched {len(basic_data)} days of historical data")
        print(f"Latest close price: ${basic_data['Close'].iloc[-1]:.2f}")
    else:
        print("Failed to fetch basic historical data")
        return
    
    print("\n=== AVAILABLE FEATURES ===")
    for key, value in FEATURE_MAP.items():
        print(f"{key:2d}: {value}")
    
    print()
    feature_request = get_feature_request("Enter feature number: ")
    
    feature_data = process_feature_request(symbol, feature_request)
    
    if feature_data is not None:
        print(f"\n✅ Successfully processed {FEATURE_MAP[feature_request]} for {symbol}")
    else:
        print(f"\n❌ Failed to process {FEATURE_MAP[feature_request]} for {symbol}")
    
    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
    

