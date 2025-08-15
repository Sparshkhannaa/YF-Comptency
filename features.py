import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class Feature:
    def __init__(self, name, description, data_type, feature_id):
        self.name = name
        self.description = description
        self.data_type = data_type
        self.feature_id = feature_id
        self.data = None
        
    def process_feature(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            
            if self.feature_id == 1:
                self.data = self._process_history(ticker)
            elif self.feature_id == 2:
                self.data = self._process_info(ticker)
            elif self.feature_id == 3:
                self.data = self._process_actions(ticker)
            elif self.feature_id == 4:
                self.data = self._process_dividends(ticker)
            elif self.feature_id == 5:
                self.data = self._process_splits(ticker)
            elif self.feature_id == 6:
                self.data = self._process_financials(ticker)
            elif self.feature_id == 7:
                self.data = self._process_balance_sheet(ticker)
            elif self.feature_id == 8:
                self.data = self._process_cashflow(ticker)
            elif self.feature_id == 9:
                self.data = self._process_earnings(ticker)
            elif self.feature_id == 10:
                self.data = self._process_quarterly_earnings(ticker)
            elif self.feature_id == 11:
                self.data = self._process_major_holders(ticker)
            elif self.feature_id == 12:
                self.data = self._process_institutional_holders(ticker)
            elif self.feature_id == 13:
                self.data = self._process_mutualfund_holders(ticker)
            elif self.feature_id == 14:
                self.data = self._process_options(ticker)
            elif self.feature_id == 15:
                self.data = self._process_option_chain(ticker)
            elif self.feature_id == 16:
                self.data = self._process_news(ticker)
            elif self.feature_id == 17:
                self.data = self._process_recommendations(ticker)
            elif self.feature_id == 18:
                self.data = self._process_sustainability(ticker)
            else:
                raise ValueError(f"Invalid feature ID: {self.feature_id}")
                
            return self.data
            
        except Exception as e:
            print(f"Error processing feature {self.feature_id}: {str(e)}")
            return None
    
    def _process_history(self, ticker):
        hist = ticker.history(period="max")
        if not hist.empty:
            hist['Returns'] = hist['Close'].pct_change()
            hist['Volatility'] = hist['Returns'].rolling(window=20).std()
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        return hist
    
    def _process_info(self, ticker):
        info = ticker.info
        key_metrics = {
            'Company Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'Forward P/E': info.get('forwardPE', 'N/A'),
            'PEG Ratio': info.get('pegRatio', 'N/A'),
            'Price to Book': info.get('priceToBook', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A')
        }
        return key_metrics
    
    def _process_actions(self, ticker):
        actions = ticker.actions
        if not actions.empty:
            actions['Action_Type'] = 'Dividend'
            splits = ticker.splits
            if not splits.empty:
                splits['Action_Type'] = 'Split'
                actions = pd.concat([actions, splits])
        return actions
    
    def _process_dividends(self, ticker):
        dividends = ticker.dividends
        if not dividends.empty:
            dividends_df = pd.DataFrame(dividends)
            dividends_df['Year'] = dividends_df.index.year
            dividends_df['Month'] = dividends_df.index.month
            dividends_df['Annual_Total'] = dividends_df.groupby('Year')['Dividends'].transform('sum')
        return dividends_df if not dividends.empty else dividends
    
    def _process_splits(self, ticker):
        splits = ticker.splits
        if not splits.empty:
            splits_df = pd.DataFrame(splits)
            splits_df['Split_Ratio'] = splits_df['Stock Splits']
            splits_df['Date'] = splits_df.index
        return splits_df if not splits.empty else splits
    
    def _process_financials(self, ticker):
        financials = ticker.financials
        if not financials.empty:
            if 'Total Revenue' in financials.index:
                financials.loc['Profit_Margin'] = (financials.loc['Net Income'] / financials.loc['Total Revenue']) * 100
        return financials
    
    def _process_balance_sheet(self, ticker):
        balance_sheet = ticker.balance_sheet
        if not balance_sheet.empty:
            if 'Total Assets' in balance_sheet.index and 'Total Liab' in balance_sheet.index:
                balance_sheet.loc['Debt_to_Assets'] = balance_sheet.loc['Total Liab'] / balance_sheet.loc['Total Assets']
        return balance_sheet
    
    def _process_cashflow(self, ticker):
        cashflow = ticker.cashflow
        if not cashflow.empty:
            if 'Operating Cash Flow' in cashflow.index and 'Capital Expenditure' in cashflow.index:
                cashflow.loc['Free_Cash_Flow'] = cashflow.loc['Operating Cash Flow'] + cashflow.loc['Capital Expenditure']
        return cashflow
    
    def _process_earnings(self, ticker):
        earnings = ticker.earnings
        if not earnings.empty:
            earnings['Earnings_Per_Share'] = earnings['Earnings']
            earnings['Revenue_Growth'] = earnings['Revenue'].pct_change() * 100
        return earnings
    
    def _process_quarterly_earnings(self, ticker):
        quarterly_earnings = ticker.quarterly_earnings
        if not quarterly_earnings.empty:
            quarterly_earnings['EPS'] = quarterly_earnings['Earnings']
            quarterly_earnings['Revenue_Growth_QoQ'] = quarterly_earnings['Revenue'].pct_change() * 100
        return quarterly_earnings
    
    def _process_major_holders(self, ticker):
        major_holders = ticker.major_holders
        if major_holders is not None and not major_holders.empty:
            major_holders.columns = ['Percentage', 'Holder']
            major_holders['Percentage'] = major_holders['Percentage'].astype(str).str.rstrip('%').astype(float)
        return major_holders
    
    def _process_institutional_holders(self, ticker):
        institutional_holders = ticker.institutional_holders
        if institutional_holders is not None and not institutional_holders.empty:
            institutional_holders['Shares'] = institutional_holders['Shares'].astype(int)
            institutional_holders['Value'] = institutional_holders['Value'].astype(float)
        return institutional_holders
    
    def _process_mutualfund_holders(self, ticker):
        mutualfund_holders = ticker.mutualfund_holders
        if mutualfund_holders is not None and not mutualfund_holders.empty:
            mutualfund_holders['Shares'] = mutualfund_holders['Shares'].astype(int)
            mutualfund_holders['Value'] = mutualfund_holders['Value'].astype(float)
        return mutualfund_holders
    
    def _process_options(self, ticker):
        options = ticker.options
        if options:
            options_df = pd.DataFrame(options, columns=['Expiration_Date'])
            options_df['Expiration_Date'] = pd.to_datetime(options_df['Expiration_Date'])
            options_df['Days_to_Expiry'] = (options_df['Expiration_Date'] - pd.Timestamp.now()).dt.days
        return options_df if options else pd.DataFrame()
    
    def _process_option_chain(self, ticker):
        options = ticker.options
        if options:
            first_expiry = options[0]
            option_chain = ticker.option_chain(first_expiry)
            
            calls = option_chain.calls
            puts = option_chain.puts
            
            if not calls.empty:
                calls['Option_Type'] = 'Call'
            if not puts.empty:
                puts['Option_Type'] = 'Put'
                
            combined = pd.concat([calls, puts]) if not calls.empty and not puts.empty else (calls if not calls.empty else puts)
            return combined
        return pd.DataFrame()
    
    def _process_news(self, ticker):
        news = ticker.news
        if news:
            news_df = pd.DataFrame(news)
            news_df['Published_Date'] = pd.to_datetime(news_df['providerPublishTime'], unit='s')
            news_df = news_df[['title', 'link', 'Published_Date', 'publisher']]
        return news_df if news else pd.DataFrame()
    
    def _process_recommendations(self, ticker):
        recommendations = ticker.recommendations
        if not recommendations.empty:
            recommendations['Date'] = pd.to_datetime(recommendations.index)
            recommendations['Year'] = recommendations['Date'].dt.year
            recommendations['Month'] = recommendations['Date'].dt.month
        return recommendations
    
    def _process_sustainability(self, ticker):
        sustainability = ticker.sustainability
        if sustainability is not None and not sustainability.empty:
            key_metrics = {
                'ESG_Score': sustainability.get('esgScores', {}).get('totalEsg', 'N/A'),
                'Environment_Score': sustainability.get('esgScores', {}).get('environmentScore', 'N/A'),
                'Social_Score': sustainability.get('esgScores', {}).get('socialScore', 'N/A'),
                'Governance_Score': sustainability.get('esgScores', {}).get('governanceScore', 'N/A')
            }
            return key_metrics
        return {'ESG_Score': 'N/A', 'Environment_Score': 'N/A', 'Social_Score': 'N/A', 'Governance_Score': 'N/A'}
    
    def get_summary(self):
        if self.data is None:
            return "No data processed yet"
        
        if isinstance(self.data, dict):
            return f"Processed {self.name}: {len(self.data)} key metrics"
        elif isinstance(self.data, pd.DataFrame):
            return f"Processed {self.name}: {len(self.data)} rows, {len(self.data.columns)} columns"
        else:
            return f"Processed {self.name}: {type(self.data).__name__}"
    
    def display_data(self, limit=10):
        if self.data is None:
            print("No data to display")
            return
        
        print(f"\n=== {self.name.upper()} DATA ===")
        print(f"Description: {self.description}")
        print(f"Data Type: {self.data_type}")
        
        if isinstance(self.data, dict):
            for key, value in list(self.data.items())[:limit]:
                print(f"{key}: {value}")
        elif isinstance(self.data, pd.DataFrame):
            print(f"Shape: {self.data.shape}")
            print("\nFirst few rows:")
            print(self.data.head(limit))
        else:
            print(self.data)