#!/usr/bin/env python3
"""
Script to generate and visualize confusion matrices for all trained models on test set
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
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
    print(f"Class 0 (Down): {np.sum(y_test == 0):,} ({np.mean(y_test == 0)*100:.1f}%)")
    print(f"Class 1 (Up): {np.sum(y_test == 1):,} ({np.mean(y_test == 1)*100:.1f}%)")
    
    return X_test, y_test, feature_cols, test_df

def load_model_and_predict(model_path, scaler_path, X_test, model_name):
    """Load model and make predictions"""
    
    print(f"  Loading {model_name}...")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # Make predictions
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return y_pred, y_pred_proba

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot confusion matrix with detailed metrics"""
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down (0)', 'Up (1)'], 
                yticklabels=['Down (0)', 'Up (1)'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot normalized confusion matrix
    plt.subplot(2, 2, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['Down (0)', 'Up (1)'], 
                yticklabels=['Down (0)', 'Up (1)'])
    plt.title(f'{model_name} - Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot metrics
    plt.subplot(2, 2, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    values = [accuracy, precision, recall, specificity, f1]
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    bars = plt.bar(metrics, values, color=colors)
    plt.title(f'{model_name} - Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot detailed confusion matrix breakdown
    plt.subplot(2, 2, 4)
    breakdown_text = f"""
Confusion Matrix Breakdown:
True Negatives (TN): {tn:,}
False Positives (FP): {fp:,}
False Negatives (FN): {fn:,}
True Positives (TP): {tp:,}

Total Samples: {tn + fp + fn + tp:,}
Correct Predictions: {tp + tn:,}
Incorrect Predictions: {fp + fn:,}

Class Distribution:
Actual Down: {tn + fn:,} ({((tn + fn)/(tn + fp + fn + tp))*100:.1f}%)
Actual Up: {tp + fp:,} ({((tp + fp)/(tn + fp + fn + tp))*100:.1f}%)
    """
    
    plt.text(0.1, 0.5, breakdown_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    plt.axis('off')
    plt.title(f'{model_name} - Detailed Breakdown')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, accuracy, precision, recall, specificity, f1

def generate_all_confusion_matrices():
    """Generate confusion matrices for all models"""
    
    print("Generating Confusion Matrices for All Models")
    print("=" * 60)
    
    # Load test data
    X_test, y_test, feature_cols, test_df = load_test_data()
    
    # Create output directory
    output_dir = './confusion_matrices'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define models to test
    models_to_test = [
        ('./saved_models/fold_final/Logistic_Regression', 'Logistic Regression'),
        ('./saved_models/fold_final/Random_Forest', 'Random Forest'),
        ('./saved_models/fold_final/Gradient_Boosted_Trees', 'Gradient Boosted Trees')
    ]
    
    all_results = {}
    
    for model_dir, model_name in models_to_test:
        print(f"\nProcessing {model_name}")
        print("-" * 40)
        
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        if os.path.exists(model_path):
            # Load model and make predictions
            y_pred, y_pred_proba = load_model_and_predict(model_path, scaler_path, X_test, model_name)
            
            # Generate confusion matrix plot
            save_path = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_confusion_matrix.png')
            cm, accuracy, precision, recall, specificity, f1 = plot_confusion_matrix(
                y_test, y_pred, model_name, save_path
            )
            
            # Store results
            all_results[model_name] = {
                'confusion_matrix': cm,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Print detailed classification report
            print(f"\n  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
            
        else:
            print(f"  Model not found: {model_path}")
    
    return all_results, test_df

def create_comparison_plot(all_results, save_path='./confusion_matrices/model_comparison.png'):
    """Create a comparison plot of all models"""
    
    print(f"\nCreating Model Comparison Plot")
    print("-" * 40)
    
    # Extract metrics for comparison
    models = list(all_results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    
    # Create comparison data
    comparison_data = []
    for model in models:
        for metric in metrics:
            comparison_data.append({
                'Model': model,
                'Metric': metric,
                'Score': all_results[model][metric.lower().replace('-', '_')]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Bar chart comparison
    plt.subplot(2, 2, 1)
    pivot_df = comparison_df.pivot(index='Model', columns='Metric', values='Score')
    pivot_df.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Confusion matrices side by side
    plt.subplot(2, 2, 2)
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    if len(models) == 1:
        axes = [axes]
    
    for i, (model, results) in enumerate(all_results.items()):
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        axes[i].set_title(f'{model}\nAccuracy: {results["accuracy"]:.3f}')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to: {save_path}")
    plt.show()

def save_confusion_matrix_results(all_results, test_df, output_file='./confusion_matrices/confusion_matrix_results.txt'):
    """Save confusion matrix results to file"""
    
    with open(output_file, 'w') as f:
        f.write("Confusion Matrix Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Data Summary:\n")
        f.write(f"  Total samples: {len(test_df):,}\n")
        f.write(f"  Date range: {test_df['Date'].min()} to {test_df['Date'].max()}\n")
        f.write(f"  Class distribution: {np.bincount(test_df['Target'])}\n")
        f.write(f"  Down class: {np.sum(test_df['Target'] == 0):,} ({np.mean(test_df['Target'] == 0)*100:.1f}%)\n")
        f.write(f"  Up class: {np.sum(test_df['Target'] == 1):,} ({np.mean(test_df['Target'] == 1)*100:.1f}%)\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"{model_name} Results:\n")
            f.write("-" * 30 + "\n")
            
            cm = results['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            
            f.write(f"Confusion Matrix:\n")
            f.write(f"  [[{tn:4d} {fp:4d}]\n")
            f.write(f"   [{fn:4d} {tp:4d}]]\n\n")
            
            f.write(f"Metrics:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  Specificity: {results['specificity']:.4f}\n")
            f.write(f"  F1-Score:  {results['f1_score']:.4f}\n\n")
            
            f.write(f"Detailed Breakdown:\n")
            f.write(f"  True Negatives (TN): {tn:,}\n")
            f.write(f"  False Positives (FP): {fp:,}\n")
            f.write(f"  False Negatives (FN): {fn:,}\n")
            f.write(f"  True Positives (TP): {tp:,}\n\n")
    
    print(f"Confusion matrix results saved to: {output_file}")

def main():
    """Main function to run confusion matrix analysis"""
    
    print("Stock Movement Prediction - Confusion Matrix Analysis")
    print("=" * 70)
    
    # Check if models exist
    if not os.path.exists('./saved_models'):
        print("Error: No saved models found. Please run train_models.py first.")
        return
    
    # Generate confusion matrices
    all_results, test_df = generate_all_confusion_matrices()
    
    if all_results:
        # Create comparison plot
        create_comparison_plot(all_results)
        
        # Save results
        save_confusion_matrix_results(all_results, test_df)
        
        print(f"\nConfusion matrix analysis completed!")
        print(f"Results saved to ./confusion_matrices/ directory")
    else:
        print("No models found to analyze.")

if __name__ == "__main__":
    main()
