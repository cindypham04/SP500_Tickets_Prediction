#!/usr/bin/env python3
"""
Script to add lag features to the existing technical features dataset
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

def add_lag_features_to_dataset(input_file, output_file):
    """Add lag features to existing dataset"""
    
    print("🔍 Loading existing technical features dataset...")
    
    # Load the existing dataset
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    print(f"📊 Loaded dataset with shape: {df.shape}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"🏢 Number of stocks: {df['Ticker'].nunique()}")
    
    # Check if Return column exists and is log returns
    if 'Return' in df.columns:
        print("✅ Log returns already present in dataset")
        print(f"📈 Return statistics: mean={df['Return'].mean():.6f}, std={df['Return'].std():.6f}")
    else:
        print("⚠️  No Return column found, calculating log returns...")
        # Sort by ticker and date first
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Calculate log returns for each stock
        def calculate_log_returns(group):
            group['Return'] = np.log(group['Close'] / group['Close'].shift(1))
            return group
        
        df = df.groupby('Ticker', group_keys=False).apply(calculate_log_returns)
        print("✅ Log returns calculated")
    
    # Sort by ticker and date
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    print("\n🔄 Adding lag features...")
    
    # Group by ticker to add lag features within each stock
    def add_lag_features_to_stock(group):
        """Add lag features for a single stock"""
        
        # Sort by date to ensure proper order
        group = group.sort_values('Date').reset_index(drop=True)
        
        # Add lag features: returns from the last N days (5 and 22 as requested)
        group['Lag_Return_1'] = group['Return'].shift(1)
        group['Lag_Return_2'] = group['Return'].shift(2)
        group['Lag_Return_3'] = group['Return'].shift(3)
        group['Lag_Return_4'] = group['Return'].shift(4)
        group['Lag_Return_5'] = group['Return'].shift(5)
        
        # Additional lag features for 22 days
        group['Lag_Return_10'] = group['Return'].shift(10)
        group['Lag_Return_15'] = group['Return'].shift(15)
        group['Lag_Return_20'] = group['Return'].shift(20)
        group['Lag_Return_22'] = group['Return'].shift(22)
        
        # Lag return statistics (mean and std over rolling windows)
        group['Lag_Return_Mean_5'] = group['Return'].shift(1).rolling(5).mean()
        group['Lag_Return_Mean_22'] = group['Return'].shift(1).rolling(22).mean()
        group['Lag_Return_Std_5'] = group['Return'].shift(1).rolling(5).std()
        group['Lag_Return_Std_22'] = group['Return'].shift(1).rolling(22).std()
        
        # Additional lag features for price ratios
        group['Lag_Price_MA5_Ratio_1'] = group['Price_MA5_Ratio'].shift(1)
        group['Lag_Price_MA5_Ratio_5'] = group['Price_MA5_Ratio'].shift(5)
        group['Lag_Price_MA20_Ratio_1'] = group['Price_MA20_Ratio'].shift(1)
        group['Lag_Price_MA20_Ratio_5'] = group['Price_MA20_Ratio'].shift(5)
        
        # Lag volume features
        group['Lag_Volume_Ratio_5_1'] = group['Volume_Ratio_5'].shift(1)
        group['Lag_Volume_Ratio_5_5'] = group['Volume_Ratio_5'].shift(5)
        
        # Lag volatility features
        group['Lag_Volatility_5_1'] = group['Volatility_5'].shift(1)
        group['Lag_Volatility_20_1'] = group['Volatility_20'].shift(1)
        
        return group
    
    # Apply lag features to each stock
    print("Processing each stock to add lag features...")
    df_with_lags = df.groupby('Ticker', group_keys=False).apply(add_lag_features_to_stock)
    
    # Clean data - replace inf and NaN values
    print("\n🧹 Cleaning data...")
    df_with_lags = df_with_lags.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0 for lag features (they will be NaN for the first few days)
    lag_columns = [col for col in df_with_lags.columns if col.startswith('Lag_')]
    df_with_lags[lag_columns] = df_with_lags[lag_columns].fillna(0)
    
    # Fill any remaining NaN values
    df_with_lags = df_with_lags.fillna(0)
    
    # Remove rows where we don't have enough history for lag features
    # Keep rows where at least the 5-day lag features are available
    df_with_lags = df_with_lags.dropna(subset=['Lag_Return_5']).reset_index(drop=True)
    
    print(f"\n📊 Final dataset shape: {df_with_lags.shape}")
    
    # Count new lag features
    original_features = len([col for col in df.columns if col not in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']])
    new_features = len([col for col in df_with_lags.columns if col not in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']])
    lag_features_added = new_features - original_features
    
    print(f"🔢 Added {lag_features_added} lag features")
    print(f"📈 Total features now: {new_features}")
    
    # Save the enhanced dataset
    df_with_lags.to_csv(output_file, index=False)
    print(f"💾 Enhanced dataset with lag features saved to: {output_file}")
    
    # Show sample of new lag features
    print(f"\n📋 Sample lag features:")
    lag_sample_cols = ['Ticker', 'Date', 'Return', 'Lag_Return_1', 'Lag_Return_5', 'Lag_Return_22', 'Lag_Return_Mean_5', 'Lag_Return_Mean_22']
    available_lag_cols = [col for col in lag_sample_cols if col in df_with_lags.columns]
    print(df_with_lags[available_lag_cols].head(10))
    
    # Show all lag features created
    print(f"\n🔢 All lag features created:")
    for i, feature in enumerate(lag_columns, 1):
        print(f"{i:2d}. {feature}")
    
    # Show summary statistics
    print(f"\n📊 Enhanced Dataset Summary:")
    print(f"   • Total records: {len(df_with_lags):,}")
    print(f"   • Number of stocks: {df_with_lags['Ticker'].nunique()}")
    print(f"   • Date range: {df_with_lags['Date'].min()} to {df_with_lags['Date'].max()}")
    print(f"   • Average records per stock: {len(df_with_lags) / df_with_lags['Ticker'].nunique():.1f}")
    print(f"   • Total features: {new_features}")
    print(f"   • Lag features added: {lag_features_added}")
    print(f"   • Target distribution: {df_with_lags['Target'].value_counts().to_dict()}")
    
    return df_with_lags

if __name__ == "__main__":
    input_file = "./sp500_daily_technical_features.csv"
    output_file = "./sp500_daily_features_with_lags.csv"
    
    print("🚀 Adding lag features to existing technical features dataset...")
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output file: {output_file}")
    
    try:
        enhanced_df = add_lag_features_to_dataset(input_file, output_file)
        print("\n✅ Successfully added lag features!")
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found.")
        print("Please make sure the technical features dataset exists first.")
    except Exception as e:
        print(f"❌ Error: {e}")
