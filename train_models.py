#!/usr/bin/env python3
"""
Script to train and evaluate Logistic Regression, Random Forest, and Gradient Boosted Trees models
using sequential validation splits for stock movement prediction
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def save_models(models_dict, fold_num, output_dir='./saved_models'):
    """Save trained models to disk"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"    Saving models for fold {fold_num}...")
    
    for model_name, model_data in models_dict.items():
        # Create model-specific directory
        model_dir = os.path.join(output_dir, f'fold_{fold_num}', model_name.replace(' ', '_'))
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_dir, 'model.joblib')
        joblib.dump(model_data['model'], model_file)
        
        # Save scaler if it exists (for Logistic Regression)
        if model_data['scaler'] is not None:
            scaler_file = os.path.join(model_dir, 'scaler.joblib')
            joblib.dump(model_data['scaler'], scaler_file)
        
        # Save metrics
        metrics_file = os.path.join(model_dir, 'metrics.joblib')
        metrics = {
            'accuracy': model_data['accuracy'],
            'precision': model_data['precision'],
            'recall': model_data['recall'],
            'f1': model_data['f1'],
            'auc': model_data['auc']
        }
        joblib.dump(metrics, metrics_file)
        
        print(f"      Saved {model_name} to {model_dir}")
    
    print(f"    All models for fold {fold_num} saved successfully!")

def load_model(model_path):
    """Load a saved model"""
    return joblib.load(model_path)

def load_fold_data(fold_dir, fold_num):
    """Load training and validation data for a specific fold"""
    
    print(f"  Loading fold {fold_num} data...")
    train_file = os.path.join(fold_dir, f'fold_{fold_num}_train.csv')
    val_file = os.path.join(fold_dir, f'fold_{fold_num}_validation.csv')
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Fold {fold_num} files not found")
    
    print(f"    Reading training file: {train_file}")
    train_df = pd.read_csv(train_file, parse_dates=['Date'])
    print(f"    Reading validation file: {val_file}")
    val_df = pd.read_csv(val_file, parse_dates=['Date'])
    
    print(f"    Training data loaded: {train_df.shape}")
    print(f"    Validation data loaded: {val_df.shape}")
    
    return train_df, val_df

def prepare_features_and_target(df):
    """Prepare features and target for training"""
    
    print(f"    Preparing features and target...")
    
    # Get feature columns (exclude metadata and target)
    exclude_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"    Found {len(feature_cols)} feature columns")
    print(f"    Data shape before processing: {df.shape}")
    
    X = df[feature_cols].values
    y = df['Target'].values
    
    print(f"    Features matrix shape: {X.shape}")
    print(f"    Target vector shape: {y.shape}")
    print(f"    Target distribution: {np.bincount(y)}")
    
    return X, y, feature_cols

def train_and_evaluate_models(train_df, val_df, fold_num):
    """Train and evaluate both models on a single fold"""
    
    print(f"\nFold {fold_num} - Training and Evaluation")
    print("-" * 40)
    
    # Prepare data
    print(f"  Preparing training data...")
    X_train, y_train, feature_cols = prepare_features_and_target(train_df)
    print(f"  Preparing validation data...")
    X_val, y_val, _ = prepare_features_and_target(val_df)
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Features: {len(feature_cols)}")
    
    # Handle any remaining NaN values
    print(f"  Cleaning data (handling NaN/inf values)...")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Data cleaning completed")
    
    # Scale features for Logistic Regression
    print(f"  Scaling features for Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print(f"  Feature scaling completed")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        ),
        'Gradient Boosted Trees': xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            scale_pos_weight=1,  # Handle class imbalance
            n_jobs=-1,
            eval_metric='logloss'
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        # Train model
        if model_name == 'Logistic Regression':
            print(f"    Fitting Logistic Regression on scaled data...")
            model.fit(X_train_scaled, y_train)
            print(f"    Making predictions on validation set...")
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:  # Random Forest or XGBoost
            print(f"    Fitting {model_name} on original data...")
            model.fit(X_train, y_train)
            print(f"    Making predictions on validation set...")
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        print(f"    Calculating metrics...")
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': model,
            'scaler': scaler if model_name == 'Logistic Regression' else None
        }
        
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        print(f"    AUC: {auc:.4f}")
        print(f"    {model_name} training completed!")
    
    # Save models for this fold
    save_models(results, fold_num)
    
    return results

def run_sequential_validation(fold_dir='./sequential_splits_v1', n_folds=5):
    """Run sequential validation across all folds"""
    
    print("Sequential Validation for Stock Movement Prediction")
    print("=" * 60)
    print(f"Processing {n_folds} folds from directory: {fold_dir}")
    
    all_results = {}
    
    for fold_num in range(1, n_folds + 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING FOLD {fold_num}/{n_folds}")
        print(f"{'='*60}")
        
        try:
            # Load fold data
            train_df, val_df = load_fold_data(fold_dir, fold_num)
            
            # Train and evaluate models
            fold_results = train_and_evaluate_models(train_df, val_df, fold_num)
            
            # Store results
            print(f"\n  Storing results for fold {fold_num}...")
            for model_name, metrics in fold_results.items():
                if model_name not in all_results:
                    all_results[model_name] = {
                        'accuracies': [],
                        'precisions': [],
                        'recalls': [],
                        'f1_scores': [],
                        'aucs': []
                    }
                
                all_results[model_name]['accuracies'].append(metrics['accuracy'])
                all_results[model_name]['precisions'].append(metrics['precision'])
                all_results[model_name]['recalls'].append(metrics['recall'])
                all_results[model_name]['f1_scores'].append(metrics['f1'])
                all_results[model_name]['aucs'].append(metrics['auc'])
            
            print(f"  Fold {fold_num} completed successfully!")
        
        except FileNotFoundError as e:
            print(f"  Skipping fold {fold_num}: {e}")
            continue
        except Exception as e:
            print(f"  Error in fold {fold_num}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL VALIDATION COMPLETED")
    print(f"{'='*60}")
    
    return all_results

def evaluate_on_test_set(fold_dir='./sequential_splits_v1', test_file='./sequential_splits_v1/final_test.csv'):
    """Evaluate best models on final test set"""
    
    print(f"\n{'='*60}")
    print(f"FINAL TEST SET EVALUATION")
    print(f"{'='*60}")
    
    # Load test data
    print(f"Loading test data from: {test_file}")
    test_df = pd.read_csv(test_file, parse_dates=['Date'])
    print(f"Test data loaded: {test_df.shape}")
    
    X_test, y_test, feature_cols = prepare_features_and_target(test_df)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Test samples: {len(X_test):,}")
    
    # Train final models on all training data (last fold)
    print(f"\nLoading final training data (fold 5)...")
    train_df, _ = load_fold_data(fold_dir, 5)  # Use last fold's training data
    X_train, y_train, _ = prepare_features_and_target(train_df)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    print(f"Scaling features for final models...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Feature scaling completed")
    
    # Train final models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        'Gradient Boosted Trees': xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1, n_jobs=-1, eval_metric='logloss')
    }
    
    test_results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining final {model_name} on all training data...")
        
        # Train model
        if model_name == 'Logistic Regression':
            print(f"  Fitting Logistic Regression on scaled data...")
            model.fit(X_train_scaled, y_train)
            print(f"  Making predictions on test set...")
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:  # Random Forest or XGBoost
            print(f"  Fitting {model_name} on original data...")
            model.fit(X_train, y_train)
            print(f"  Making predictions on test set...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"  Calculating test metrics...")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        test_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Precision: {precision:.4f}")
        print(f"  Test Recall: {recall:.4f}")
        print(f"  Test F1-Score: {f1:.4f}")
        print(f"  Test AUC: {auc:.4f}")
        print(f"  {model_name} test evaluation completed!")
    
    # Save final models
    print(f"\nSaving final models...")
    final_models_dict = {}
    for model_name, model in models.items():
        final_models_dict[model_name] = {
            'model': model,
            'scaler': scaler if model_name == 'Logistic Regression' else None,
            'accuracy': test_results[model_name]['accuracy'],
            'precision': test_results[model_name]['precision'],
            'recall': test_results[model_name]['recall'],
            'f1': test_results[model_name]['f1'],
            'auc': test_results[model_name]['auc']
        }
    
    save_models(final_models_dict, 'final', './saved_models')
    
    return test_results

def print_summary_results(validation_results, test_results):
    """Print summary of all results"""
    
    print(f"\nSequential Validation Summary")
    print("=" * 50)
    
    # Calculate and display key metrics first
    print(f"\nKEY METRICS SUMMARY:")
    print("-" * 30)
    
    model_summary = []
    for model_name in validation_results.keys():
        avg_val_auc = np.mean(validation_results[model_name]['aucs'])
        test_auc = test_results[model_name]['auc'] if model_name in test_results else 0.0
        model_summary.append((model_name, avg_val_auc, test_auc))
    
    # Sort by average validation AUC
    model_summary.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<25} {'Avg Val AUC':<12} {'Test AUC':<10}")
    print("-" * 50)
    for model_name, avg_val_auc, test_auc in model_summary:
        print(f"{model_name:<25} {avg_val_auc:<12.4f} {test_auc:<10.4f}")
    
    # Detailed results for each model
    for model_name in validation_results.keys():
        print(f"\n{model_name}:")
        print("-" * len(model_name))
        
        # Validation results (mean ± std)
        val_acc = np.mean(validation_results[model_name]['accuracies'])
        val_acc_std = np.std(validation_results[model_name]['accuracies'])
        val_prec = np.mean(validation_results[model_name]['precisions'])
        val_prec_std = np.std(validation_results[model_name]['precisions'])
        val_rec = np.mean(validation_results[model_name]['recalls'])
        val_rec_std = np.std(validation_results[model_name]['recalls'])
        val_f1 = np.mean(validation_results[model_name]['f1_scores'])
        val_f1_std = np.std(validation_results[model_name]['f1_scores'])
        val_auc = np.mean(validation_results[model_name]['aucs'])
        val_auc_std = np.std(validation_results[model_name]['aucs'])
        
        print(f"   Validation Accuracy:  {val_acc:.4f} ± {val_acc_std:.4f}")
        print(f"   Validation Precision: {val_prec:.4f} ± {val_prec_std:.4f}")
        print(f"   Validation Recall:    {val_rec:.4f} ± {val_rec_std:.4f}")
        print(f"   Validation F1-Score:  {val_f1:.4f} ± {val_f1_std:.4f}")
        print(f"   Validation AUC:       {val_auc:.4f} ± {val_auc_std:.4f}")
        
        # Test results
        if model_name in test_results:
            print(f"   Test Accuracy:       {test_results[model_name]['accuracy']:.4f}")
            print(f"   Test Precision:      {test_results[model_name]['precision']:.4f}")
            print(f"   Test Recall:         {test_results[model_name]['recall']:.4f}")
            print(f"   Test F1-Score:       {test_results[model_name]['f1']:.4f}")
            print(f"   Test AUC:            {test_results[model_name]['auc']:.4f}")

def create_model_summary(output_dir='./saved_models'):
    """Create a summary file of all saved models"""
    
    summary_file = os.path.join(output_dir, 'model_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("Stock Movement Prediction - Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # List all saved models
        if os.path.exists(output_dir):
            for fold_dir in sorted(os.listdir(output_dir)):
                if fold_dir.startswith('fold_'):
                    f.write(f"Fold {fold_dir}:\n")
                    fold_path = os.path.join(output_dir, fold_dir)
                    
                    for model_dir in sorted(os.listdir(fold_path)):
                        model_path = os.path.join(fold_path, model_dir)
                        metrics_file = os.path.join(model_path, 'metrics.joblib')
                        
                        if os.path.exists(metrics_file):
                            metrics = joblib.load(metrics_file)
                            f.write(f"  {model_dir.replace('_', ' ')}:\n")
                            f.write(f"    Accuracy: {metrics['accuracy']:.4f}\n")
                            f.write(f"    Precision: {metrics['precision']:.4f}\n")
                            f.write(f"    Recall: {metrics['recall']:.4f}\n")
                            f.write(f"    F1-Score: {metrics['f1']:.4f}\n")
                            f.write(f"    AUC: {metrics['auc']:.4f}\n\n")
        
        f.write("Final Models:\n")
        final_path = os.path.join(output_dir, 'final')
        if os.path.exists(final_path):
            for model_dir in sorted(os.listdir(final_path)):
                model_path = os.path.join(final_path, model_dir)
                metrics_file = os.path.join(model_path, 'metrics.joblib')
                
                if os.path.exists(metrics_file):
                    metrics = joblib.load(metrics_file)
                    f.write(f"  {model_dir.replace('_', ' ')}:\n")
                    f.write(f"    Test Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"    Test Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Test Recall: {metrics['recall']:.4f}\n")
                    f.write(f"    Test F1-Score: {metrics['f1']:.4f}\n")
                    f.write(f"    Test AUC: {metrics['auc']:.4f}\n\n")
    
    print(f"Model summary saved to: {summary_file}")

def main():
    """Main function to run sequential validation"""
    
    fold_dir = './sequential_splits_v1'
    test_file = './sequential_splits_v1/final_test.csv'
    
    # Check if files exist
    if not os.path.exists(fold_dir):
        print(f"Error: Sequential splits directory '{fold_dir}' not found.")
        print("Please run sequential_validation.py first.")
        return
    
    if not os.path.exists(test_file):
        print(f"Error: Test file '{test_file}' not found.")
        return
    
    # Run sequential validation
    validation_results = run_sequential_validation(fold_dir)
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(fold_dir, test_file)
    
    # Print summary
    print_summary_results(validation_results, test_results)
    
    # Create model summary file
    create_model_summary()
    
    print(f"\nSequential validation completed")
    print(f"Results show performance across multiple time periods")
    print(f"Test results represent final out-of-sample performance")
    print(f"All models saved to ./saved_models/ directory")

if __name__ == "__main__":
    main()
