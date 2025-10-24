#!/usr/bin/env python3
"""
Script to implement Sequential Validation for time series stock prediction
Based on the paper: "Predicting stock movement direction with machine learning"

Sequential Validation: Multiple rolling time-ordered splits that respect temporal order
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def create_sequential_validation_splits(input_file, output_dir='./sequential_splits', n_splits=5):
    """
    Create sequential validation splits using rolling time-ordered windows
    
    Parameters:
    - n_splits: Number of sequential folds (default 5)
    - Each fold uses expanding training window with fixed validation period
    """
    
    print("Loading dataset...")
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Number of stocks: {df['Ticker'].nunique()}")
    
    # Sort by date to ensure chronological order
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Get unique dates
    unique_dates = sorted(df['Date'].unique())
    total_days = len(unique_dates)
    
    print(f"Total unique trading days: {total_days}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate split parameters
    # Reserve last 20% for final test set
    test_size = int(total_days * 0.2)
    train_val_days = total_days - test_size
    
    # For sequential validation, use expanding windows
    fold_size = train_val_days // n_splits
    
    print(f"\nSequential Validation Setup:")
    print(f"   Number of folds: {n_splits}")
    print(f"   Days per fold: {fold_size}")
    print(f"   Final test period: {test_size} days")
    
    splits_info = []
    
    for fold in range(n_splits):
        print(f"\nCreating Fold {fold + 1}/{n_splits}...")
        
        # Calculate date ranges for this fold
        train_end_idx = (fold + 1) * fold_size
        val_start_idx = train_end_idx
        val_end_idx = train_end_idx + fold_size
        
        # Get date boundaries
        train_end_date = unique_dates[train_end_idx - 1] if train_end_idx <= len(unique_dates) else unique_dates[-1]
        val_start_date = unique_dates[val_start_idx] if val_start_idx < len(unique_dates) else unique_dates[-1]
        val_end_date = unique_dates[val_end_idx - 1] if val_end_idx <= len(unique_dates) else unique_dates[-1]
        
        # Create splits
        train_df = df[df['Date'] <= train_end_date].copy()
        val_df = df[(df['Date'] >= val_start_date) & (df['Date'] <= val_end_date)].copy()
        
        # Skip if validation set is too small
        if len(val_df) < 1000:  # Minimum validation samples
            print(f"   Skipping fold {fold + 1} - validation set too small ({len(val_df)} samples)")
            continue
        
        # Save fold data
        train_file = os.path.join(output_dir, f'fold_{fold+1}_train.csv')
        val_file = os.path.join(output_dir, f'fold_{fold+1}_validation.csv')
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        
        # Record split information
        split_info = {
            'fold': fold + 1,
            'train_start': train_df['Date'].min(),
            'train_end': train_df['Date'].max(),
            'val_start': val_df['Date'].min(),
            'val_end': val_df['Date'].max(),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'train_stocks': train_df['Ticker'].nunique(),
            'val_stocks': val_df['Ticker'].nunique()
        }
        splits_info.append(split_info)
        
        print(f"   Training: {split_info['train_start']} to {split_info['train_end']} ({split_info['train_samples']:,} samples)")
        print(f"   Validation: {split_info['val_start']} to {split_info['val_end']} ({split_info['val_samples']:,} samples)")
    
    # Create final test set (last 20% of data)
    print(f"\nCreating final test set...")
    test_start_date = unique_dates[-test_size]
    test_df = df[df['Date'] >= test_start_date].copy()
    
    test_file = os.path.join(output_dir, 'final_test.csv')
    test_df.to_csv(test_file, index=False)
    
    print(f"   Test: {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df):,} samples)")
    
    # Save splits summary
    splits_summary = pd.DataFrame(splits_info)
    summary_file = os.path.join(output_dir, 'splits_summary.csv')
    splits_summary.to_csv(summary_file, index=False)
    
    print(f"\nSequential validation splits created successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Splits summary saved to: {summary_file}")
    
    return splits_info, test_df

def create_alternative_sequential_splits(input_file, output_dir='./sequential_splits_v2', 
                                       train_years=6, val_months=6, n_splits=5):
    """
    Alternative sequential validation with fixed training years and validation months
    
    Parameters:
    - train_years: Years of training data (expanding window)
    - val_months: Months of validation data per fold
    - n_splits: Number of sequential folds
    """
    
    print("Creating alternative sequential validation splits...")
    print(f"Training window: {train_years} years")
    print(f"Validation window: {val_months} months per fold")
    print(f"Number of folds: {n_splits}")
    
    df = pd.read_csv(input_file, parse_dates=['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Get date boundaries
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # Calculate fold parameters
    fold_duration = pd.DateOffset(months=val_months)
    
    os.makedirs(output_dir, exist_ok=True)
    
    splits_info = []
    
    for fold in range(n_splits):
        print(f"\nCreating Fold {fold + 1}/{n_splits}...")
        
        # Calculate validation period
        val_start = start_date + pd.DateOffset(years=train_years) + (fold * fold_duration)
        val_end = val_start + fold_duration - pd.DateOffset(days=1)
        
        # Skip if validation period extends beyond available data
        if val_end > end_date:
            print(f"   Skipping fold {fold + 1} - validation period beyond available data")
            continue
        
        # Create splits
        train_df = df[df['Date'] < val_start].copy()
        val_df = df[(df['Date'] >= val_start) & (df['Date'] <= val_end)].copy()
        
        # Skip if validation set is too small
        if len(val_df) < 1000:
            print(f"   Skipping fold {fold + 1} - validation set too small ({len(val_df)} samples)")
            continue
        
        # Save fold data
        train_file = os.path.join(output_dir, f'fold_{fold+1}_train.csv')
        val_file = os.path.join(output_dir, f'fold_{fold+1}_validation.csv')
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        
        # Record split information
        split_info = {
            'fold': fold + 1,
            'train_start': train_df['Date'].min(),
            'train_end': train_df['Date'].max(),
            'val_start': val_df['Date'].min(),
            'val_end': val_df['Date'].max(),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'train_stocks': train_df['Ticker'].nunique(),
            'val_stocks': val_df['Ticker'].nunique()
        }
        splits_info.append(split_info)
        
        print(f"   Training: {split_info['train_start']} to {split_info['train_end']} ({split_info['train_samples']:,} samples)")
        print(f"   Validation: {split_info['val_start']} to {split_info['val_end']} ({split_info['val_samples']:,} samples)")
    
    # Save splits summary
    splits_summary = pd.DataFrame(splits_info)
    summary_file = os.path.join(output_dir, 'splits_summary.csv')
    splits_summary.to_csv(summary_file, index=False)
    
    print(f"\nAlternative sequential validation splits created!")
    print(f"Output directory: {output_dir}")
    
    return splits_info

def main():
    """Main function to create sequential validation splits"""
    
    print("Sequential Validation for Time Series Stock Prediction")
    print("=" * 60)
    
    input_file = "./sp500_daily_features_with_indices.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    print(f"Input file: {input_file}")
    
    # Method 1: Equal fold sizes
    print("\nMethod 1: Equal Fold Sizes")
    print("-" * 30)
    splits_info1, test_df = create_sequential_validation_splits(
        input_file, 
        output_dir='./sequential_splits_v1',
        n_splits=5
    )
    
    # Method 2: Fixed training years + validation months
    print("\nMethod 2: Fixed Training Years + Validation Months")
    print("-" * 50)
    splits_info2 = create_alternative_sequential_splits(
        input_file,
        output_dir='./sequential_splits_v2',
        train_years=6,
        val_months=6,
        n_splits=5
    )
    
    # Summary
    print(f"\nSequential Validation Summary:")
    print(f"   Method 1 folds: {len(splits_info1)}")
    print(f"   Method 2 folds: {len(splits_info2)}")
    print(f"   Final test samples: {len(test_df):,}")
    
    # Show feature information
    feature_columns = [col for col in test_df.columns if col not in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']]
    print(f"\nFeatures for training:")
    print(f"   Total features: {len(feature_columns)}")
    print(f"   Technical indicators: {len([col for col in feature_columns if not col.startswith('Lag_') and not any(idx in col for idx in ['Nikkei', 'Hang_Seng', 'All_Ordinaries', 'DAX', 'FTSE', 'NYSE', 'DJIA'])])}")
    print(f"   Lag features: {len([col for col in feature_columns if col.startswith('Lag_')])}")
    print(f"   International indices: {len([col for col in feature_columns if any(idx in col for idx in ['Nikkei', 'Hang_Seng', 'All_Ordinaries', 'DAX', 'FTSE', 'NYSE', 'DJIA'])])}")
    
    print(f"\nSequential validation setup completed!")
    print(f"Use these splits for training and validation, then test on final_test.csv")

if __name__ == "__main__":
    main()
