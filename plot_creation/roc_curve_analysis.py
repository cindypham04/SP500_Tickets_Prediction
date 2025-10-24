#!/usr/bin/env python3
"""
Script to generate ROC curves for all trained models on test set
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
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
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return y_pred_proba

def plot_roc_curves(all_results, y_test, save_path='./roc_curves/roc_curves_comparison.png'):
    """Plot ROC curves for all models"""
    
    print(f"\nGenerating ROC Curves")
    print("-" * 40)
    
    # Create output directory
    os.makedirs('./roc_curves', exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Plot ROC curve for each model
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (model_name, y_pred_proba) in enumerate(all_results.items()):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        print(f"  {model_name}: AUC = {roc_auc:.4f}")
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
             label='Random Classifier (AUC = 0.500)')
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves Comparison - Stock Movement Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text box with interpretation
    textstr = '''ROC Curve Interpretation:
• Closer to top-left corner = Better performance
• AUC = 1.0: Perfect classifier
• AUC = 0.5: Random classifier
• AUC > 0.7: Good performance
• AUC > 0.8: Excellent performance'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves comparison saved to: {save_path}")
    plt.show()
    
    return save_path

def plot_individual_roc_curves(all_results, y_test):
    """Plot individual ROC curves for each model"""
    
    print(f"\nGenerating Individual ROC Curves")
    print("-" * 40)
    
    for model_name, y_pred_proba in all_results.items():
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create individual plot
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='blue', lw=3, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
                 label='Random Classifier (AUC = 0.500)')
        
        # Customize the plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add performance interpretation
        if roc_auc >= 0.8:
            performance = "Excellent"
            color = "green"
        elif roc_auc >= 0.7:
            performance = "Good"
            color = "orange"
        elif roc_auc >= 0.6:
            performance = "Fair"
            color = "yellow"
        else:
            performance = "Poor"
            color = "red"
        
        textstr = f'''Performance: {performance}
AUC Score: {roc_auc:.3f}

Threshold Analysis:
• Optimal threshold maximizes (TPR - FPR)
• Higher threshold = More conservative predictions
• Lower threshold = More aggressive predictions'''
        
        props = dict(boxstyle='round', facecolor=color, alpha=0.3)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        # Save individual plot
        save_path = f'./roc_curves/{model_name.replace(" ", "_")}_roc_curve.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  {model_name} ROC curve saved to: {save_path}")
        plt.show()

def analyze_optimal_thresholds(all_results, y_test):
    """Analyze optimal thresholds for each model"""
    
    print(f"\nOptimal Threshold Analysis")
    print("-" * 40)
    
    threshold_results = {}
    
    for model_name, y_pred_proba in all_results.items():
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        # Find optimal threshold (maximizes TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred_optimal)
        precision = precision_score(y_test, y_pred_optimal)
        recall = recall_score(y_test, y_pred_optimal)
        f1 = f1_score(y_test, y_pred_optimal)
        
        threshold_results[model_name] = {
            'optimal_threshold': optimal_threshold,
            'fpr': optimal_fpr,
            'tpr': optimal_tpr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"  {model_name}:")
        print(f"    Optimal Threshold: {optimal_threshold:.4f}")
        print(f"    TPR: {optimal_tpr:.4f}, FPR: {optimal_fpr:.4f}")
        print(f"    Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    return threshold_results

def plot_threshold_analysis(all_results, y_test, save_path='./roc_curves/threshold_analysis.png'):
    """Plot threshold analysis for all models"""
    
    print(f"\nGenerating Threshold Analysis Plot")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: ROC curves with optimal points
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green']
    
    for i, (model_name, y_pred_proba) in enumerate(all_results.items()):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Mark optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', color=colors[i], 
                markersize=8, label=f'{model_name} Optimal')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves with Optimal Thresholds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Threshold vs Performance
    ax2 = axes[0, 1]
    threshold_results = analyze_optimal_thresholds(all_results, y_test)
    
    models = list(threshold_results.keys())
    accuracies = [threshold_results[model]['accuracy'] for model in models]
    precisions = [threshold_results[model]['precision'] for model in models]
    recalls = [threshold_results[model]['recall'] for model in models]
    f1_scores = [threshold_results[model]['f1_score'] for model in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    ax2.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax2.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax2.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax2.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Score')
    ax2.set_title('Performance at Optimal Thresholds')
    ax2.set_xticks(x)
    ax2.set_xticklabels([model.replace(' ', '\n') for model in models])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall curves
    ax3 = axes[1, 0]
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    for i, (model_name, y_pred_proba) in enumerate(all_results.items()):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        ax3.plot(recall, precision, color=colors[i], lw=2,
                label=f'{model_name} (AP = {avg_precision:.3f})')
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: AUC comparison
    ax4 = axes[1, 1]
    auc_scores = []
    for model_name, y_pred_proba in all_results.items():
        auc_score = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc_score)
    
    bars = ax4.bar(models, auc_scores, color=colors[:len(models)], alpha=0.7)
    ax4.set_ylabel('AUC Score')
    ax4.set_title('AUC Score Comparison')
    ax4.set_xticklabels([model.replace(' ', '\n') for model in models], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Threshold analysis plot saved to: {save_path}")
    plt.show()

def save_roc_results(all_results, y_test, threshold_results, output_file='./roc_curves/roc_results.txt'):
    """Save ROC analysis results to file"""
    
    with open(output_file, 'w') as f:
        f.write("ROC Curve Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Data Summary:\n")
        f.write(f"  Total samples: {len(y_test):,}\n")
        f.write(f"  Positive class (Up): {np.sum(y_test):,} ({np.mean(y_test)*100:.1f}%)\n")
        f.write(f"  Negative class (Down): {len(y_test) - np.sum(y_test):,} ({(1-np.mean(y_test))*100:.1f}%)\n\n")
        
        for model_name, y_pred_proba in all_results.items():
            f.write(f"{model_name} ROC Analysis:\n")
            f.write("-" * 30 + "\n")
            
            # Calculate ROC metrics
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            f.write(f"AUC Score: {roc_auc:.4f}\n")
            f.write(f"Performance: {'Excellent' if roc_auc >= 0.8 else 'Good' if roc_auc >= 0.7 else 'Fair' if roc_auc >= 0.6 else 'Poor'}\n\n")
            
            if model_name in threshold_results:
                tr = threshold_results[model_name]
                f.write(f"Optimal Threshold Analysis:\n")
                f.write(f"  Optimal Threshold: {tr['optimal_threshold']:.4f}\n")
                f.write(f"  True Positive Rate: {tr['tpr']:.4f}\n")
                f.write(f"  False Positive Rate: {tr['fpr']:.4f}\n")
                f.write(f"  Accuracy: {tr['accuracy']:.4f}\n")
                f.write(f"  Precision: {tr['precision']:.4f}\n")
                f.write(f"  Recall: {tr['recall']:.4f}\n")
                f.write(f"  F1-Score: {tr['f1_score']:.4f}\n\n")
    
    print(f"ROC results saved to: {output_file}")

def main():
    """Main function to run ROC curve analysis"""
    
    print("Stock Movement Prediction - ROC Curve Analysis")
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
        # Generate ROC curves
        plot_roc_curves(all_results, y_test)
        
        # Generate individual ROC curves
        plot_individual_roc_curves(all_results, y_test)
        
        # Analyze optimal thresholds
        threshold_results = analyze_optimal_thresholds(all_results, y_test)
        
        # Generate threshold analysis plot
        plot_threshold_analysis(all_results, y_test)
        
        # Save results
        save_roc_results(all_results, y_test, threshold_results)
        
        print(f"\nROC curve analysis completed!")
        print(f"Results saved to ./roc_curves/ directory")
    else:
        print("No models found to analyze.")

if __name__ == "__main__":
    main()
