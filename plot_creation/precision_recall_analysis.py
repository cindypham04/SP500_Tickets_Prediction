#!/usr/bin/env python3
"""
Script to generate Precision-Recall curves for all trained models on test set
Precision-Recall curves are especially useful for imbalanced datasets
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
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
    
    # Calculate class imbalance
    imbalance_ratio = np.sum(y_test == 1) / np.sum(y_test == 0)
    print(f"Class imbalance ratio (Up/Down): {imbalance_ratio:.3f}")
    
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
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return y_pred_proba

def plot_precision_recall_curves(all_results, y_test, save_path='./precision_recall/precision_recall_comparison.png'):
    """Plot Precision-Recall curves for all models"""
    
    print(f"\nGenerating Precision-Recall Curves")
    print("-" * 40)
    
    # Create output directory
    os.makedirs('./precision_recall', exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Plot Precision-Recall curve for each model
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (model_name, y_pred_proba) in enumerate(all_results.items()):
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{model_name} (AP = {avg_precision:.3f})')
        
        print(f"  {model_name}: Average Precision = {avg_precision:.4f}")
    
    # Plot baseline (random classifier)
    baseline_precision = np.sum(y_test) / len(y_test)  # Proportion of positive class
    plt.axhline(y=baseline_precision, color='gray', linestyle='--', lw=1,
               label=f'Random Classifier (AP = {baseline_precision:.3f})')
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison - Stock Movement Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text box with interpretation
    textstr = '''Precision-Recall Interpretation:
• Higher curve = Better performance
• AP = 1.0: Perfect classifier
• AP = baseline: Random classifier
• AP > 0.7: Good performance
• AP > 0.8: Excellent performance

For imbalanced datasets, PR curves
are more informative than ROC curves'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curves comparison saved to: {save_path}")
    plt.show()
    
    return save_path

def plot_individual_precision_recall_curves(all_results, y_test):
    """Plot individual Precision-Recall curves for each model"""
    
    print(f"\nGenerating Individual Precision-Recall Curves")
    print("-" * 40)
    
    baseline_precision = np.sum(y_test) / len(y_test)
    
    for model_name, y_pred_proba in all_results.items():
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Create individual plot
        plt.figure(figsize=(10, 8))
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, color='blue', lw=3, 
                label=f'{model_name} (AP = {avg_precision:.3f})')
        
        # Plot baseline
        plt.axhline(y=baseline_precision, color='gray', linestyle='--', lw=1,
                   label=f'Random Classifier (AP = {baseline_precision:.3f})')
        
        # Customize the plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add performance interpretation
        if avg_precision >= 0.8:
            performance = "Excellent"
            color = "green"
        elif avg_precision >= 0.7:
            performance = "Good"
            color = "orange"
        elif avg_precision >= 0.6:
            performance = "Fair"
            color = "yellow"
        elif avg_precision >= baseline_precision + 0.1:
            performance = "Better than Random"
            color = "lightblue"
        else:
            performance = "Poor"
            color = "red"
        
        textstr = f'''Performance: {performance}
Average Precision: {avg_precision:.3f}
Baseline (Random): {baseline_precision:.3f}

Threshold Analysis:
• Higher threshold = Higher precision, Lower recall
• Lower threshold = Lower precision, Higher recall
• Optimal threshold balances precision & recall'''
        
        props = dict(boxstyle='round', facecolor=color, alpha=0.3)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        # Save individual plot
        save_path = f'./precision_recall/{model_name.replace(" ", "_")}_precision_recall.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  {model_name} Precision-Recall curve saved to: {save_path}")
        plt.show()

def analyze_optimal_precision_recall_thresholds(all_results, y_test):
    """Analyze optimal thresholds for precision-recall trade-off"""
    
    print(f"\nOptimal Precision-Recall Threshold Analysis")
    print("-" * 40)
    
    threshold_results = {}
    
    for model_name, y_pred_proba in all_results.items():
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Find optimal threshold using F1-score
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred_optimal)
        precision_at_threshold = precision_score(y_test, y_pred_optimal)
        recall_at_threshold = recall_score(y_test, y_pred_optimal)
        f1_at_threshold = f1_score(y_test, y_pred_optimal)
        
        threshold_results[model_name] = {
            'optimal_threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1_score': optimal_f1,
            'accuracy': accuracy,
            'precision_at_threshold': precision_at_threshold,
            'recall_at_threshold': recall_at_threshold,
            'f1_at_threshold': f1_at_threshold
        }
        
        print(f"  {model_name}:")
        print(f"    Optimal Threshold: {optimal_threshold:.4f}")
        print(f"    Precision: {optimal_precision:.4f}, Recall: {optimal_recall:.4f}")
        print(f"    F1-Score: {optimal_f1:.4f}")
        print(f"    Accuracy at threshold: {accuracy:.4f}")
    
    return threshold_results

def plot_precision_recall_threshold_analysis(all_results, y_test, save_path='./precision_recall/threshold_analysis.png'):
    """Plot comprehensive threshold analysis"""
    
    print(f"\nGenerating Precision-Recall Threshold Analysis Plot")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Precision-Recall curves with optimal points
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green']
    baseline_precision = np.sum(y_test) / len(y_test)
    
    for i, (model_name, y_pred_proba) in enumerate(all_results.items()):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        ax1.plot(recall, precision, color=colors[i], lw=2, 
                label=f'{model_name} (AP = {avg_precision:.3f})')
        
        # Mark optimal threshold (F1-score)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        ax1.plot(recall[optimal_idx], precision[optimal_idx], 'o', color=colors[i], 
                markersize=8, label=f'{model_name} Optimal')
    
    ax1.axhline(y=baseline_precision, color='gray', linestyle='--', lw=1, label='Random Classifier')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves with Optimal Thresholds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Precision comparison
    ax2 = axes[0, 1]
    models = list(all_results.keys())
    avg_precisions = [average_precision_score(y_test, y_pred_proba) for y_pred_proba in all_results.values()]
    
    bars = ax2.bar(models, avg_precisions, color=colors[:len(models)], alpha=0.7)
    ax2.axhline(y=baseline_precision, color='gray', linestyle='--', lw=2, label=f'Random Baseline ({baseline_precision:.3f})')
    ax2.set_ylabel('Average Precision')
    ax2.set_title('Average Precision Comparison')
    ax2.set_xticklabels([model.replace(' ', '\n') for model in models], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_precisions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 3: Precision vs Recall scatter
    ax3 = axes[1, 0]
    threshold_results = analyze_optimal_precision_recall_thresholds(all_results, y_test)
    
    for i, (model_name, results) in enumerate(threshold_results.items()):
        ax3.scatter(results['recall'], results['precision'], 
                   color=colors[i], s=100, label=model_name, alpha=0.7)
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall at Optimal Thresholds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: F1-Score comparison
    ax4 = axes[1, 1]
    f1_scores = [results['f1_score'] for results in threshold_results.values()]
    
    bars = ax4.bar(models, f1_scores, color=colors[:len(models)], alpha=0.7)
    ax4.set_ylabel('F1-Score')
    ax4.set_title('F1-Score at Optimal Thresholds')
    ax4.set_xticklabels([model.replace(' ', '\n') for model in models], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall threshold analysis plot saved to: {save_path}")
    plt.show()

def plot_precision_recall_vs_threshold(all_results, y_test, save_path='./precision_recall/precision_recall_vs_threshold.png'):
    """Plot precision and recall vs threshold for all models"""
    
    print(f"\nGenerating Precision-Recall vs Threshold Plot")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green']
    
    for i, (model_name, y_pred_proba) in enumerate(all_results.items()):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Plot precision and recall vs threshold
        axes[i].plot(thresholds, precision[:-1], 'b-', label='Precision', linewidth=2)
        axes[i].plot(thresholds, recall[:-1], 'r-', label='Recall', linewidth=2)
        
        # Find optimal threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Mark optimal threshold
        axes[i].axvline(x=optimal_threshold, color='green', linestyle='--', 
                       label=f'Optimal Threshold ({optimal_threshold:.3f})')
        
        axes[i].set_xlabel('Threshold')
        axes[i].set_ylabel('Score')
        axes[i].set_title(f'{model_name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall vs threshold plot saved to: {save_path}")
    plt.show()

def save_precision_recall_results(all_results, y_test, threshold_results, output_file='./precision_recall/precision_recall_results.txt'):
    """Save precision-recall analysis results to file"""
    
    baseline_precision = np.sum(y_test) / len(y_test)
    
    with open(output_file, 'w') as f:
        f.write("Precision-Recall Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Data Summary:\n")
        f.write(f"  Total samples: {len(y_test):,}\n")
        f.write(f"  Positive class (Up): {np.sum(y_test):,} ({np.mean(y_test)*100:.1f}%)\n")
        f.write(f"  Negative class (Down): {len(y_test) - np.sum(y_test):,} ({(1-np.mean(y_test))*100:.1f}%)\n")
        f.write(f"  Class imbalance ratio: {np.sum(y_test) / (len(y_test) - np.sum(y_test)):.3f}\n")
        f.write(f"  Random baseline precision: {baseline_precision:.3f}\n\n")
        
        for model_name, y_pred_proba in all_results.items():
            f.write(f"{model_name} Precision-Recall Analysis:\n")
            f.write("-" * 30 + "\n")
            
            # Calculate metrics
            avg_precision = average_precision_score(y_test, y_pred_proba)
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"Performance vs Random: {avg_precision - baseline_precision:.4f}\n")
            f.write(f"Performance Level: {'Excellent' if avg_precision >= 0.8 else 'Good' if avg_precision >= 0.7 else 'Fair' if avg_precision >= 0.6 else 'Better than Random' if avg_precision >= baseline_precision + 0.1 else 'Poor'}\n\n")
            
            if model_name in threshold_results:
                tr = threshold_results[model_name]
                f.write(f"Optimal Threshold Analysis (F1-Score):\n")
                f.write(f"  Optimal Threshold: {tr['optimal_threshold']:.4f}\n")
                f.write(f"  Precision: {tr['precision']:.4f}\n")
                f.write(f"  Recall: {tr['recall']:.4f}\n")
                f.write(f"  F1-Score: {tr['f1_score']:.4f}\n")
                f.write(f"  Accuracy: {tr['accuracy']:.4f}\n\n")
    
    print(f"Precision-Recall results saved to: {output_file}")

def main():
    """Main function to run precision-recall analysis"""
    
    print("Stock Movement Prediction - Precision-Recall Analysis")
    print("=" * 70)
    
    # Check if models exist
    if not os.path.exists('./saved_models'):
        print("Error: No saved models found. Please run train_models.py first.")
        return
    
    # Load test data
    X_test, y_test, feature_cols, test_df = load_test_data()
    
    # Define models to test
    models_to_test = [
        ('./saved_models/fold_final/Logistic_Regression', 'Logistic Regression'),
        ('./saved_models/fold_final/Random_Forest', 'Random Forest'),
        ('./saved_models/fold_final/Gradient_Boosted_Trees', 'Gradient Boosted Trees')
    ]
    
    all_results = {}
    
    # Load models and get predictions
    for model_dir, model_name in models_to_test:
        print(f"\nProcessing {model_name}")
        print("-" * 40)
        
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        if os.path.exists(model_path):
            y_pred_proba = load_model_and_predict(model_path, scaler_path, X_test, model_name)
            all_results[model_name] = y_pred_proba
        else:
            print(f"  Model not found: {model_path}")
    
    if all_results:
        # Generate Precision-Recall curves
        plot_precision_recall_curves(all_results, y_test)
        
        # Generate individual Precision-Recall curves
        plot_individual_precision_recall_curves(all_results, y_test)
        
        # Analyze optimal thresholds
        threshold_results = analyze_optimal_precision_recall_thresholds(all_results, y_test)
        
        # Generate threshold analysis plot
        plot_precision_recall_threshold_analysis(all_results, y_test)
        
        # Generate precision-recall vs threshold plot
        plot_precision_recall_vs_threshold(all_results, y_test)
        
        # Save results
        save_precision_recall_results(all_results, y_test, threshold_results)
        
        print(f"\nPrecision-Recall analysis completed!")
        print(f"Results saved to ./precision_recall/ directory")
    else:
        print("No models found to analyze.")

if __name__ == "__main__":
    main()
