#!/usr/bin/env python3
"""
Script to download international market indices and add them to the existing dataset
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def download_indices_data(start_date='2018-01-01', end_date=None):
    """Download international market indices data"""
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("🌍 Downloading international market indices...")
    print(f"📅 Date range: {start_date} to {end_date}")
    
    # Define the indices with their Yahoo Finance symbols
    indices = {
        # Asian markets
        'Nikkei_225': '^N225',           # Japan
        'Hang_Seng': '^HSI',             # Hong Kong
        'All_Ordinaries': '^AORD',       # Australia
        
        # European markets
        'DAX': '^GDAXI',                 # Germany
        'FTSE_100': '^FTSE',             # UK
        
        # US markets
        'NYSE_Composite': '^NYA',        # NYSE Composite
        'DJIA': '^DJI'                   # Dow Jones Industrial Average
    }
    
    all_indices_data = {}
    
    for name, symbol in indices.items():
        print(f"📈 Downloading {name} ({symbol})...")
        
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                # Reset index to get Date as column
                data = data.reset_index()
                
                # Convert Date to datetime and remove timezone info
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
                
                # Rename columns to avoid conflicts
                data = data.rename(columns={
                    'Date': 'Date',
                    'Open': f'{name}_Open',
                    'High': f'{name}_High', 
                    'Low': f'{name}_Low',
                    'Close': f'{name}_Close',
                    'Volume': f'{name}_Volume'
                })
                
                # Calculate returns
                data[f'{name}_Return'] = np.log(data[f'{name}_Close'] / data[f'{name}_Close'].shift(1))
                
                # Calculate additional features
                data[f'{name}_MA_5'] = data[f'{name}_Close'].rolling(5).mean()
                data[f'{name}_MA_20'] = data[f'{name}_Close'].rolling(20).mean()
                data[f'{name}_Volatility_20'] = data[f'{name}_Return'].rolling(20).std()
                data[f'{name}_Price_MA20_Ratio'] = data[f'{name}_Close'] / data[f'{name}_MA_20']
                
                # Keep only relevant columns
                keep_cols = ['Date', f'{name}_Close', f'{name}_Return', f'{name}_MA_5', 
                           f'{name}_MA_20', f'{name}_Volatility_20', f'{name}_Price_MA20_Ratio']
                data = data[keep_cols]
                
                all_indices_data[name] = data
                print(f"✅ {name}: {len(data)} records")
            else:
                print(f"⚠️  {name}: No data downloaded")
                
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            continue
    
    return all_indices_data

def merge_indices_with_stock_data(stock_file, indices_data, output_file):
    """Merge indices data with stock data"""
    
    print(f"\n🔗 Loading stock data from {stock_file}...")
    stock_df = pd.read_csv(stock_file, parse_dates=['Date'])
    
    print(f"📊 Stock data shape: {stock_df.shape}")
    print(f"📅 Stock date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
    
    # Start with stock data
    merged_df = stock_df.copy()
    
    # Merge each index
    for name, index_data in indices_data.items():
        print(f"🔄 Merging {name} data...")
        
        # Convert Date to datetime if not already
        index_data['Date'] = pd.to_datetime(index_data['Date'])
        
        # Merge on Date
        merged_df = pd.merge(merged_df, index_data, on='Date', how='left')
        
        print(f"✅ {name} merged: {merged_df.shape[1]} columns")
    
    # Fill NaN values for indices (weekends, holidays)
    print("\n🧹 Cleaning data...")
    
    # Forward fill missing values for indices (use last available value)
    index_columns = [col for col in merged_df.columns if any(idx in col for idx in ['Nikkei', 'Hang_Seng', 'All_Ordinaries', 'DAX', 'FTSE', 'NYSE', 'DJIA'])]
    
    for col in index_columns:
        merged_df[col] = merged_df[col].fillna(method='ffill')
    
    # Fill any remaining NaN with 0
    merged_df = merged_df.fillna(0)
    
    # Replace inf values
    merged_df = merged_df.replace([np.inf, -np.inf], 0)
    
    print(f"📊 Final merged dataset shape: {merged_df.shape}")
    
    # Count new features
    original_features = len([col for col in stock_df.columns if col not in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']])
    new_features = len([col for col in merged_df.columns if col not in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']])
    indices_features_added = new_features - original_features
    
    print(f"🔢 Added {indices_features_added} international indices features")
    print(f"📈 Total features now: {new_features}")
    
    # Save the enhanced dataset
    merged_df.to_csv(output_file, index=False)
    print(f"💾 Enhanced dataset with international indices saved to: {output_file}")
    
    # Show sample of new features
    print(f"\n📋 Sample international indices features:")
    sample_cols = ['Ticker', 'Date', 'Return', 'Nikkei_225_Return', 'Hang_Seng_Return', 'DAX_Return', 'FTSE_100_Return', 'DJIA_Return']
    available_cols = [col for col in sample_cols if col in merged_df.columns]
    print(merged_df[available_cols].head(10))
    
    # Show all indices features created
    print(f"\n🌍 All international indices features created:")
    indices_features = [col for col in merged_df.columns if any(idx in col for idx in ['Nikkei', 'Hang_Seng', 'All_Ordinaries', 'DAX', 'FTSE', 'NYSE', 'DJIA'])]
    for i, feature in enumerate(indices_features, 1):
        print(f"{i:2d}. {feature}")
    
    # Show summary statistics
    print(f"\n📊 Enhanced Dataset Summary:")
    print(f"   • Total records: {len(merged_df):,}")
    print(f"   • Number of stocks: {merged_df['Ticker'].nunique()}")
    print(f"   • Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"   • Average records per stock: {len(merged_df) / merged_df['Ticker'].nunique():.1f}")
    print(f"   • Total features: {new_features}")
    print(f"   • International indices features: {indices_features_added}")
    print(f"   • Target distribution: {merged_df['Target'].value_counts().to_dict()}")
    
    return merged_df

def main():
    """Main function to download indices and merge with stock data"""
    
    print("🚀 Adding international market indices to stock dataset...")
    
    # File paths
    stock_file = "./sp500_daily_features_with_lags.csv"
    output_file = "./sp500_daily_features_with_indices.csv"
    
    print(f"📁 Input file: {stock_file}")
    print(f"📁 Output file: {output_file}")
    
    try:
        # Download indices data
        indices_data = download_indices_data()
        
        if not indices_data:
            print("❌ No indices data downloaded. Exiting.")
            return
        
        print(f"\n✅ Successfully downloaded {len(indices_data)} indices")
        
        # Merge with stock data
        enhanced_df = merge_indices_with_stock_data(stock_file, indices_data, output_file)
        
        print("\n🎉 Successfully added international indices features!")
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{stock_file}' not found.")
        print("Please make sure the dataset with lag features exists first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
