#!/usr/bin/env python3
"""
Script to load trained models and evaluate them on the test set
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_test_data(test_file='./sequential_splits_v1/final_test.csv'):
    """Load and prepare test data"""
    
    print("Loading Test Data")
    print("=" * 30)
    
    # Load test data
    print(f"Reading test file: {test_file}")
    test_df = pd.read_csv(test_file, parse_dates=['Date'])
    print(f"Test data loaded: {test_df.shape}")
    
    # Prepare features and target
    print("Preparing features and target...")
    exclude_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['Target'].values
    
    # Clean data
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Target distribution: {np.bincount(y_test)}")
    
    return X_test, y_test, feature_cols, test_df

def load_and_test_model(model_path, scaler_path, X_test, y_test, model_name):
    """Load a specific model and test it"""
    
    print(f"\nTesting {model_name}")
    print("-" * 40)
    
    # Load model and scaler
    print(f"  Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        print(f"  Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
    
    # Prepare data for prediction
    if scaler is not None:
        print(f"  Scaling features...")
        X_test_scaled = scaler.transform(X_test)
        print(f"  Making predictions...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        print(f"  Making predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print(f"  Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"  Results:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    AUC:       {auc:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      [[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"       [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }

def test_final_models():
    """Test the final models (trained on all training data)"""
    
    print("Testing Final Models")
    print("=" * 50)
    
    # Load test data
    X_test, y_test, feature_cols, test_df = load_test_data()
    
    # Define model paths
    models_dir = './saved_models/final'
    models_to_test = [
        ('Logistic_Regression', 'Logistic Regression'),
        ('Random_Forest', 'Random Forest'),
        ('Gradient_Boosted_Trees', 'Gradient Boosted Trees')
    ]
    
    results = {}
    
    for model_folder, model_name in models_to_test:
        model_path = os.path.join(models_dir, model_folder, 'model.joblib')
        scaler_path = os.path.join(models_dir, model_folder, 'scaler.joblib')
        
        if os.path.exists(model_path):
            result = load_and_test_model(model_path, scaler_path, X_test, y_test, model_name)
            results[model_name] = result
        else:
            print(f"  Model not found: {model_path}")
    
    return results, test_df

def test_fold_models(fold_num):
    """Test models from a specific fold"""
    
    print(f"Testing Fold {fold_num} Models")
    print("=" * 50)
    
    # Load test data
    X_test, y_test, feature_cols, test_df = load_test_data()
    
    # Define model paths
    models_dir = f'./saved_models/fold_{fold_num}'
    models_to_test = [
        ('Logistic_Regression', 'Logistic Regression'),
        ('Random_Forest', 'Random Forest'),
        ('Gradient_Boosted_Trees', 'Gradient Boosted Trees')
    ]
    
    results = {}
    
    for model_folder, model_name in models_to_test:
        model_path = os.path.join(models_dir, model_folder, 'model.joblib')
        scaler_path = os.path.join(models_dir, model_folder, 'scaler.joblib')
        
        if os.path.exists(model_path):
            result = load_and_test_model(model_path, scaler_path, X_test, y_test, model_name)
            results[model_name] = result
        else:
            print(f"  Model not found: {model_path}")
    
    return results, test_df

def compare_all_folds():
    """Compare performance across all folds"""
    
    print("Comparing Performance Across All Folds")
    print("=" * 60)
    
    # Load test data once
    X_test, y_test, feature_cols, test_df = load_test_data()
    
    all_results = {}
    
    # Test final models
    print(f"\nTesting Final Models:")
    final_results, _ = test_final_models()
    all_results['Final'] = final_results
    
    # Test each fold
    for fold_num in range(1, 6):
        print(f"\nTesting Fold {fold_num} Models:")
        fold_results, _ = test_fold_models(fold_num)
        all_results[f'Fold_{fold_num}'] = fold_results
    
    return all_results, test_df

def print_comparison_summary(all_results):
    """Print a comparison summary of all results"""
    
    print(f"\nPerformance Comparison Summary")
    print("=" * 60)
    
    # Create comparison table
    model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosted Trees']
    
    print(f"\nAUC Comparison:")
    print(f"{'Model':<25} {'Final':<8} {'Fold 1':<8} {'Fold 2':<8} {'Fold 3':<8} {'Fold 4':<8} {'Fold 5':<8}")
    print("-" * 80)
    
    for model_name in model_names:
        row = f"{model_name:<25}"
        for fold_key in ['Final', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5']:
            if fold_key in all_results and model_name in all_results[fold_key]:
                auc = all_results[fold_key][model_name]['auc']
                row += f" {auc:<7.4f}"
            else:
                row += f" {'N/A':<7}"
        print(row)
    
    print(f"\nAccuracy Comparison:")
    print(f"{'Model':<25} {'Final':<8} {'Fold 1':<8} {'Fold 2':<8} {'Fold 3':<8} {'Fold 4':<8} {'Fold 5':<8}")
    print("-" * 80)
    
    for model_name in model_names:
        row = f"{model_name:<25}"
        for fold_key in ['Final', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5']:
            if fold_key in all_results and model_name in all_results[fold_key]:
                acc = all_results[fold_key][model_name]['accuracy']
                row += f" {acc:<7.4f}"
            else:
                row += f" {'N/A':<7}"
        print(row)

def save_test_results(all_results, test_df, output_file='test_results.txt'):
    """Save test results to a file"""
    
    with open(output_file, 'w') as f:
        f.write("Model Testing Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Data Info:\n")
        f.write(f"  Total samples: {len(test_df):,}\n")
        f.write(f"  Date range: {test_df['Date'].min()} to {test_df['Date'].max()}\n")
        f.write(f"  Target distribution: {np.bincount(test_df['Target'])}\n\n")
        
        for fold_key, fold_results in all_results.items():
            f.write(f"{fold_key} Results:\n")
            f.write("-" * 30 + "\n")
            
            for model_name, metrics in fold_results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
                f.write(f"  AUC:       {metrics['auc']:.4f}\n")
                f.write(f"  Confusion Matrix: [[{metrics['confusion_matrix'][0,0]:4d} {metrics['confusion_matrix'][0,1]:4d}]\n")
                f.write(f"                     [{metrics['confusion_matrix'][1,0]:4d} {metrics['confusion_matrix'][1,1]:4d}]]\n\n")
    
    print(f"Test results saved to: {output_file}")

def main():
    """Main function to run model testing"""
    
    print("Stock Movement Prediction - Model Testing")
    print("=" * 60)
    
    # Check if models exist
    if not os.path.exists('./saved_models'):
        print("Error: No saved models found. Please run train_models.py first.")
        return
    
    # Run comprehensive testing
    all_results, test_df = compare_all_folds()
    
    # Print summary
    print_comparison_summary(all_results)
    
    # Save results
    save_test_results(all_results, test_df)
    
    print(f"\nModel testing completed!")
    print(f"Results saved to test_results.txt")

if __name__ == "__main__":
    main()
