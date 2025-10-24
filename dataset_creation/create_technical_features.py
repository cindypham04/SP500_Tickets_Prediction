#!/usr/bin/env python3
"""
Script to create simple technical indicators for each day of each stock
"""

import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm

def create_technical_indicators(df):
    """Create technical indicators for a single stock"""
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate returns
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Create target (next day's return direction)
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    
    # Remove last row (can't predict next day)
    df = df.iloc[:-1].reset_index(drop=True)
    
    # Basic price features
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Range_Pct'] = df['Price_Range'] / df['Close']
    df['Open_Close_Range'] = abs(df['Open'] - df['Close']) / df['Close']
    df['High_Close_Range'] = (df['High'] - df['Close']) / df['Close']
    df['Low_Close_Range'] = (df['Close'] - df['Low']) / df['Close']
    
    # Volume features
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio_5'] = df['Volume'] / df['Volume_MA_5']
    df['Volume_Ratio_10'] = df['Volume'] / df['Volume_MA_10']
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume_MA_20']
    
    # Price moving averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    
    # Price ratios to moving averages
    df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
    df['Price_MA10_Ratio'] = df['Close'] / df['MA_10']
    df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
    df['Price_MA50_Ratio'] = df['Close'] / df['MA_50']
    
    # Moving average crossovers
    df['MA5_MA10'] = df['MA_5'] / df['MA_10']
    df['MA10_MA20'] = df['MA_10'] / df['MA_20']
    df['MA20_MA50'] = df['MA_20'] / df['MA_50']
    
    # Volatility features
    df['Volatility_5'] = df['Return'].rolling(5).std()
    df['Volatility_10'] = df['Return'].rolling(10).std()
    df['Volatility_20'] = df['Return'].rolling(20).std()
    
    # Momentum features
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # RSI (simplified)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Price position features
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Price_Position_20'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
    
    # Return features
    df['Return_MA_5'] = df['Return'].rolling(5).mean()
    df['Return_MA_10'] = df['Return'].rolling(10).mean()
    df['Return_MA_20'] = df['Return'].rolling(20).mean()
    
    # Skewness and Kurtosis
    df['Return_Skew_10'] = df['Return'].rolling(10).skew()
    df['Return_Kurt_10'] = df['Return'].rolling(10).kurt()
    
    # Autocorrelation features
    df['Autocorr_1'] = df['Return'].rolling(20).apply(lambda x: x.autocorr(lag=1))
    df['Autocorr_2'] = df['Return'].rolling(20).apply(lambda x: x.autocorr(lag=2))
    df['Autocorr_5'] = df['Return'].rolling(20).apply(lambda x: x.autocorr(lag=5))
    
    # Drop rows with NaN values (due to rolling calculations)
    df = df.dropna().reset_index(drop=True)
    
    return df

def process_all_stocks():
    """Process all stocks to create daily technical indicators"""
    
    print("🔍 Loading stock data...")
    data_dir = "./sp500_data"
    files = glob.glob(f"{data_dir}/*.csv")
    
    print(f"📊 Found {len(files)} stock files")
    
    # Process each stock
    all_daily_data = []
    successful_stocks = 0
    
    for file_path in tqdm(files, desc="Processing stocks"):
        ticker = os.path.basename(file_path).replace(".csv", "")
        
        try:
            # Load stock data
            df = pd.read_csv(file_path, parse_dates=['Date'])
            
            # Create technical indicators
            df_with_features = create_technical_indicators(df)
            
            if not df_with_features.empty:
                df_with_features['Ticker'] = ticker
                all_daily_data.append(df_with_features)
                successful_stocks += 1
                print(f"✅ {ticker}: {len(df_with_features)} daily records")
            else:
                print(f"⚠️  {ticker}: No features created")
                
        except Exception as e:
            print(f"❌ {ticker}: Error - {e}")
            continue
    
    # Combine all data
    if all_daily_data:
        final_data = pd.concat(all_daily_data, ignore_index=True)
        
        print(f"\n📊 Final dataset shape: {final_data.shape}")
        print(f"✅ Successfully processed {successful_stocks} stocks")
        
        # Clean data
        final_data = final_data.replace([np.inf, -np.inf], np.nan)
        final_data = final_data.fillna(0)
        
        # Save dataset
        output_file = "./sp500_daily_technical_features.csv"
        final_data.to_csv(output_file, index=False)
        print(f"💾 Daily technical features dataset saved to: {output_file}")
        
        # Show sample
        print(f"\n📋 Sample data:")
        sample_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']
        available_cols = [col for col in sample_cols if col in final_data.columns]
        print(final_data[available_cols].head())
        
        # Show feature columns
        feature_columns = [col for col in final_data.columns if col not in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']]
        print(f"\n🔢 Created {len(feature_columns)} technical features:")
        for i, feature in enumerate(feature_columns[:20]):  # Show first 20 features
            print(f"{i+1:3d}. {feature}")
        if len(feature_columns) > 20:
            print(f"... and {len(feature_columns) - 20} more features")
        
        # Show summary statistics
        print(f"\n📊 Dataset Summary:")
        print(f"   • Total records: {len(final_data):,}")
        print(f"   • Number of stocks: {final_data['Ticker'].nunique()}")
        print(f"   • Date range: {final_data['Date'].min()} to {final_data['Date'].max()}")
        print(f"   • Average records per stock: {len(final_data) / final_data['Ticker'].nunique():.1f}")
        print(f"   • Features: {len(feature_columns)}")
        print(f"   • Target distribution: {final_data['Target'].value_counts().to_dict()}")
        
        return final_data
    else:
        print("❌ No data processed successfully")
        return None

if __name__ == "__main__":
    process_all_stocks()
