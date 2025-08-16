import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to load .env file automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded successfully")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   .env file will not be loaded automatically")

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinnhubAnalyzer:
    """
    A comprehensive stock analysis tool using Finnhub API
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the analyzer with Finnhub API key
        
        Args:
            api_key (str, optional): Your Finnhub API key. If not provided, 
                                   will try to load from environment variables.
        """
        self.api_key = api_key or self._load_api_key()
        if not self.api_key:
            raise ValueError(
                "API key not provided and not found in environment variables. "
                "Please set FINNHUB_API_KEY or API_KEY in your .env file or "
                "pass it directly to the constructor."
            )
        
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'X-Finnhub-Token': self.api_key,
            'User-Agent': 'StockAnalyzer/1.0'
        })
    
    def _load_api_key(self) -> Optional[str]:
        """
        Load API key from environment variables
        
        Returns:
            Optional[str]: API key if found, None otherwise
        """
        # Try multiple environment variable names
        api_key = os.getenv('FINNHUB_API_KEY') or os.getenv('API_KEY')
        
        if api_key:
            print(f"✅ API key loaded from environment variables")
            return api_key
        else:
            print("⚠️  No API key found in environment variables")
            return None
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make a request to Finnhub API with error handling
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Query parameters
            
        Returns:
            Optional[Dict]: API response or None if failed
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("❌ API key is invalid or expired. Please check your API key.")
                return None
            elif response.status_code == 429:
                print("⚠️  Rate limit exceeded. Waiting 1 second...")
                time.sleep(1)
                return self._make_request(endpoint, params)
            else:
                print(f"❌ API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return None
    
    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile information
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[Dict]: Company profile data
        """
        print(f"📊 Fetching company profile for {symbol}...")
        return self._make_request('stock/profile2', {'symbol': symbol})
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a stock
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[Dict]: Quote data
        """
        print(f"💹 Fetching real-time quote for {symbol}...")
        return self._make_request('quote', {'symbol': symbol})
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical price data with fallback to yfinance
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to fetch
            
        Returns:
            Optional[pd.DataFrame]: Historical price data
        """
        print(f"📈 Fetching {days} days of historical data for {symbol}...")
        
        # Try Finnhub first
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            'symbol': symbol,
            'resolution': 'D',  # Daily data
            'from': int(start_date.timestamp()),
            'to': int(end_date.timestamp())
        }
        
        data = self._make_request('stock/candle', params)
        
        if data and data['s'] == 'ok':
            print(f"✅ Successfully fetched data from Finnhub")
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            })
            df.set_index('Date', inplace=True)
            return df
        else:
            print(f"⚠️  Finnhub historical data failed, trying yfinance fallback...")
            return self.get_historical_data_alternative(symbol, days)
    
    def get_historical_data_alternative(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical data using yfinance as fallback
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to fetch
            
        Returns:
            Optional[pd.DataFrame]: Historical price data
        """
        try:
            import yfinance as yf
            print(f"📈 Fetching {days} days of historical data for {symbol} using yfinance...")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            
            if not df.empty:
                print(f"✅ Successfully fetched data from yfinance")
                # Ensure we have the required columns
                if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns and 'Volume' in df.columns:
                    return df
                else:
                    print(f"❌ yfinance data missing required columns")
                    return None
            else:
                print(f"❌ No data found for {symbol} in yfinance")
                return None
                
        except ImportError:
            print("❌ yfinance not installed. Install with: pip install yfinance")
            print("   This is required for historical data fallback when Finnhub fails")
            return None
        except Exception as e:
            print(f"❌ Error fetching data from yfinance: {e}")
            return None
    
    def get_data_source_info(self, symbol: str) -> str:
        """
        Get information about available data sources for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            str: Information about data availability
        """
        info = f"📊 Data Source Information for {symbol}:\n"
        
        # Check Finnhub quote availability
        quote = self.get_quote(symbol)
        if quote:
            info += "✅ Finnhub: Real-time quotes available\n"
        else:
            info += "❌ Finnhub: Real-time quotes not available\n"
        
        # Check Finnhub profile availability
        profile = self.get_company_profile(symbol)
        if profile:
            info += "✅ Finnhub: Company profile available\n"
        else:
            info += "❌ Finnhub: Company profile not available\n"
        
        # Check yfinance availability
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                info += "✅ yfinance: Historical data available\n"
            else:
                info += "❌ yfinance: Historical data not available\n"
        except:
            info += "❌ yfinance: Not available (not installed)\n"
        
        return info
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if df is None or df.empty:
            return df
            
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    def get_news_sentiment(self, symbol: str, days: int = 7) -> Optional[Dict]:
        """
        Get news sentiment analysis
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to look back
            
        Returns:
            Optional[Dict]: News sentiment data
        """
        print(f"📰 Fetching news sentiment for {symbol}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            'q': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }
        
        return self._make_request('news-sentiment', params)
    
    def get_peers(self, symbol: str) -> Optional[List[str]]:
        """
        Get peer companies
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[List[str]]: List of peer symbols
        """
        print(f"👥 Fetching peer companies for {symbol}...")
        return self._make_request('stock/peers', {'symbol': symbol})
    
    def plot_price_analysis(self, df: pd.DataFrame, symbol: str):
        """
        Create comprehensive price analysis plots
        
        Args:
            df (pd.DataFrame): Price data with indicators
            symbol (str): Stock symbol
        """
        if df is None or df.empty:
            print("❌ No data to plot")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Technical Analysis for {symbol}', fontsize=16, fontweight='bold')
        
        # Price and Moving Averages
        axes[0].plot(df.index, df['Close'], label='Close Price', linewidth=2, color='#1f77b4')
        axes[0].plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7, color='#ff7f0e')
        axes[0].plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7, color='#2ca02c')
        axes[0].fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.1, label='Bollinger Bands')
        axes[0].set_title('Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MACD
        axes[1].plot(df.index, df['MACD'], label='MACD', color='#1f77b4')
        axes[1].plot(df.index, df['MACD_Signal'], label='Signal', color='#ff7f0e')
        axes[1].bar(df.index, df['MACD_Histogram'], label='Histogram', alpha=0.6, color='#2ca02c')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('MACD')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # RSI
        axes[2].plot(df.index, df['RSI'], label='RSI', color='#1f77b4')
        axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[2].axhline(y=50, color='black', linestyle='-', alpha=0.5, label='Neutral (50)')
        axes[2].set_title('Relative Strength Index (RSI)')
        axes[2].set_ylim(0, 100)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_volume_analysis(self, df: pd.DataFrame, symbol: str):
        """
        Create volume analysis plot
        
        Args:
            df (pd.DataFrame): Price data
            symbol (str): Stock symbol
        """
        if df is None or df.empty:
            print("❌ No data to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Volume Analysis for {symbol}', fontsize=16, fontweight='bold')
        
        # Price and Volume
        ax1.plot(df.index, df['Close'], color='#1f77b4', linewidth=2)
        ax1.set_title('Price Movement')
        ax1.grid(True, alpha=0.3)
        
        # Volume bars
        colors = ['green' if close > open else 'red' for close, open in zip(df['Close'], df['Open'])]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7)
        ax2.set_title('Trading Volume')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self, symbol: str, quote: Dict, profile: Dict, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive summary report
        
        Args:
            symbol (str): Stock symbol
            quote (Dict): Quote data
            profile (Dict): Company profile
            df (pd.DataFrame): Historical data
            
        Returns:
            str: Summary report
        """
        if not quote or not profile or df is None:
            return "❌ Insufficient data to generate report"
            
        report = f"""
{'='*60}
📊 STOCK ANALYSIS REPORT - {symbol.upper()}
{'='*60}

🏢 COMPANY INFORMATION:
   Name: {profile.get('name', 'N/A')}
   Industry: {profile.get('finnhubIndustry', 'N/A')}
   Country: {profile.get('country', 'N/A')}
   Market Cap: ${profile.get('marketCapitalization', 0):,.0f}M
   IPO Date: {profile.get('ipo', 'N/A')}

💹 CURRENT QUOTE:
   Current Price: ${quote.get('c', 0):.2f}
   Previous Close: ${quote.get('pc', 0):.2f}
   Change: ${quote.get('d', 0):.2f} ({quote.get('dp', 0):.2f}%)
   High: ${quote.get('h', 0):.2f}
   Low: ${quote.get('l', 0):.2f}
   Volume: {quote.get('v', 0):,}

📈 TECHNICAL INDICATORS:
   SMA 20: ${df['SMA_20'].iloc[-1]:.2f}
   SMA 50: ${df['SMA_50'].iloc[-1]:.2f}
   RSI: {df['RSI'].iloc[-1]:.1f}
   MACD: {df['MACD'].iloc[-1]:.4f}
   
   Price vs SMA 20: {'ABOVE' if quote.get('c', 0) > df['SMA_20'].iloc[-1] else 'BELOW'}
   Price vs SMA 50: {'ABOVE' if quote.get('c', 0) > df['SMA_50'].iloc[-1] else 'BELOW'}
   RSI Status: {'OVERBOUGHT' if df['RSI'].iloc[-1] > 70 else 'OVERSOLD' if df['RSI'].iloc[-1] < 30 else 'NEUTRAL'}

📊 PERFORMANCE SUMMARY:
   {days} Day Return: {((quote.get('c', 0) / df['Close'].iloc[0] - 1) * 100):.2f}%
   Volatility (Std Dev): {df['Close'].pct_change().std() * 100:.2f}%
   Average Volume: {df['Volume'].mean():,.0f}
   
{'='*60}
        """
        return report
    
    def run_complete_analysis(self, symbol: str, days: int = 30):
        """
        Run complete stock analysis
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days for historical data
        """
        print(f"\n🚀 Starting comprehensive analysis for {symbol.upper()}")
        print("="*60)
        
        # Fetch all data
        quote = self.get_quote(symbol)
        profile = self.get_company_profile(symbol)
        df = self.get_historical_data(symbol, days)
        
        if not all([quote, profile, df is not None]):
            print("❌ Failed to fetch required data. Please check your API key and symbol.")
            return
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Generate and display report
        report = self.generate_summary_report(symbol, quote, profile, df)
        print(report)
        
        # Create plots
        print("\n📊 Generating technical analysis charts...")
        self.plot_price_analysis(df, symbol)
        
        print("\n📊 Generating volume analysis...")
        self.plot_volume_analysis(df, symbol)
        
        # Additional analysis
        print("\n🔍 Fetching additional insights...")
        
        # News sentiment
        sentiment = self.get_news_sentiment(symbol, 7)
        if sentiment:
            print(f"📰 Recent news sentiment: {sentiment.get('sentiment', 'N/A')}")
        
        # Peer companies
        peers = self.get_peers(symbol)
        if peers:
            print(f"👥 Peer companies: {', '.join(peers[:5])}")  # Show first 5 peers
        
        print(f"\n✅ Analysis complete for {symbol.upper()}!")


def main():
    """
    Main function to run the stock analyzer
    """
    print("🚀 FINNHUB STOCK ANALYSIS TOOL")
    print("="*50)
    print("Get your free API key at: https://finnhub.io/")
    print()
    print("💡 This tool uses a hybrid approach:")
    print("   • Finnhub API for real-time quotes and company profiles")
    print("   • yfinance fallback for historical data when Finnhub fails")
    print("   • Automatic fallback ensures reliable data access")
    print()
    
    # Try to load API key from environment variables first
    api_key = None
    try:
        analyzer = FinnhubAnalyzer()  # Will auto-load from env vars
        api_key = analyzer.api_key
        print(f"✅ Using API key from environment variables")
    except ValueError as e:
        print(f"❌ {e}")
        print("\nPlease either:")
        print("1. Set FINNHUB_API_KEY or API_KEY in your .env file")
        print("2. Or enter your API key manually below")
        print()
        
        # Fallback to manual input
        api_key = input("Enter your Finnhub API key: ").strip()
        if not api_key:
            print("❌ API key is required!")
            return
    
    # Initialize analyzer
    try:
        analyzer = FinnhubAnalyzer(api_key)
    except ValueError as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("1. Analyze a stock")
        print("2. Check data source availability")
        print("3. Quit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "3" or choice.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        elif choice == "2":
            symbol = input("Enter stock symbol to check: ").strip().upper()
            if symbol:
                print("\n" + analyzer.get_data_source_info(symbol))
            continue
        
        elif choice == "1":
            symbol = input("Enter stock symbol: ").strip().upper()
            
            if not symbol:
                print("❌ Please enter a valid symbol!")
                continue
            
            try:
                days = int(input("Enter number of days for analysis (default 30): ") or "30")
                if days <= 0 or days > 365:
                    print("⚠️  Days should be between 1 and 365. Using default 30.")
                    days = 30
            except ValueError:
                print("⚠️  Invalid input. Using default 30 days.")
                days = 30
            
            # Run analysis
            try:
                analyzer.run_complete_analysis(symbol, days)
            except Exception as e:
                print(f"❌ Analysis failed: {str(e)}")
            
            print("\n" + "="*50)
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
