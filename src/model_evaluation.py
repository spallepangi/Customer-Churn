"""
Model Evaluation Module for Customer Churn Prediction

This module provides:
1. Comprehensive model evaluation metrics
2. Advanced visualization techniques
3. Model interpretation using SHAP
4. Business impact analysis
5. Model monitoring capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
                           classification_report, average_precision_score)
from sklearn.calibration import calibration_curve
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation and interpretation"""
    
    def __init__(self, model, X_test, y_test, feature_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or X_test.columns.tolist()
        
        # Make predictions
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Initialize SHAP explainer
        try:
            self.shap_explainer = shap.Explainer(model, X_test.sample(min(100, len(X_test))))
            self.shap_values = self.shap_explainer(X_test.sample(min(200, len(X_test))))
        except:
            print("SHAP explainer initialization failed. SHAP analysis will be skipped.")
            self.shap_explainer = None
            self.shap_values = None
    
    def calculate_basic_metrics(self):
        """Calculate basic classification metrics"""
        print("=== BASIC CLASSIFICATION METRICS ===")
        
        metrics = {
            'Accuracy': accuracy_score(self.y_test, self.y_pred),
            'Precision': precision_score(self.y_test, self.y_pred),
            'Recall': recall_score(self.y_test, self.y_pred),
            'F1-Score': f1_score(self.y_test, self.y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, self.y_pred_proba),
            'Average Precision': average_precision_score(self.y_test, self.y_pred_proba)
        }
        
        print("Classification Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric:20}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """Create enhanced confusion matrix visualization"""
        plt.figure(figsize=figsize)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations combining counts and percentages
        annotations = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                                for j in range(cm.shape[1])] 
                               for i in range(cm.shape[0])])
        
        # Create heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'],
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix with Percentages')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        print(f"\nConfusion Matrix Analysis:")
        print(f"True Negatives:  {tn:4d} | False Positives: {fp:4d}")
        print(f"False Negatives: {fn:4d} | True Positives:  {tp:4d}")
        print(f"\nSensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity:          {specificity:.4f}")
        
        return cm
    
    def plot_roc_curve(self, figsize=(8, 6), save_path=None):
        """Plot ROC curve with detailed analysis"""
        plt.figure(figsize=figsize)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # Find optimal threshold using Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Analysis')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"At optimal threshold - TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
        
        return fpr, tpr, thresholds, optimal_threshold
    
    def plot_precision_recall_curve(self, figsize=(8, 6), save_path=None):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=figsize)
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        # Plot PR curve
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AP = {avg_precision:.4f})')
        
        # Plot baseline (proportion of positive class)
        baseline = self.y_test.mean()
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Baseline (Random) = {baseline:.4f}')
        
        # Find optimal threshold for F1-score
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
        optimal_idx = np.argmax(f1_scores)
        
        if optimal_idx < len(thresholds):
            optimal_threshold = thresholds[optimal_idx]
            plt.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=8,
                    label=f'Max F1 Threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return precision, recall, thresholds
    
    def plot_probability_distribution(self, figsize=(12, 8), save_path=None):
        """Plot probability distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Probability histogram by class
        axes[0, 0].hist(self.y_pred_proba[self.y_test == 0], bins=50, alpha=0.7, 
                       label='Not Churned', color='blue', density=True)
        axes[0, 0].hist(self.y_pred_proba[self.y_test == 1], bins=50, alpha=0.7, 
                       label='Churned', color='red', density=True)
        axes[0, 0].set_xlabel('Predicted Probability')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Probability Distribution by True Class')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Calibration plot
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, self.y_pred_proba, n_bins=10)
        
        axes[0, 1].plot(mean_predicted_value, fraction_of_positives, "s-",
                       label="Model", color='blue')
        axes[0, 1].plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        axes[0, 1].set_xlabel('Mean Predicted Probability')
        axes[0, 1].set_ylabel('Fraction of Positives')
        axes[0, 1].set_title('Calibration Plot')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Threshold analysis
        thresholds = np.arange(0.1, 1.0, 0.05)
        precisions, recalls, f1_scores = [], [], []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            precisions.append(precision_score(self.y_test, y_pred_thresh))
            recalls.append(recall_score(self.y_test, y_pred_thresh))
            f1_scores.append(f1_score(self.y_test, y_pred_thresh))
        
        axes[1, 0].plot(thresholds, precisions, 'b-', label='Precision')
        axes[1, 0].plot(thresholds, recalls, 'r-', label='Recall')
        axes[1, 0].plot(thresholds, f1_scores, 'g-', label='F1-Score')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Metrics vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction confidence distribution
        confidence = np.abs(self.y_pred_proba - 0.5)
        axes[1, 1].hist(confidence, bins=30, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Prediction Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Model Confidence Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print confidence statistics
        print(f"\nPrediction Confidence Analysis:")
        print(f"Mean Confidence: {confidence.mean():.4f}")
        print(f"Median Confidence: {np.median(confidence):.4f}")
        print(f"High Confidence Predictions (>0.4): {(confidence > 0.4).mean():.2%}")
    
    def analyze_feature_importance(self, top_n=20, figsize=(12, 8), save_path=None):
        """Analyze feature importance if available"""
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return None
        
        print("=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=figsize)
        
        top_features = importance_df.head(top_n)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print(f"\nTop {min(15, len(importance_df))} Most Important Features:")
        print("-" * 50)
        for i, row in importance_df.head(15).iterrows():
            print(f"{row['feature']:30}: {row['importance']:.6f}")
        
        return importance_df
    
    def shap_analysis(self, save_plots=True):
        """Comprehensive SHAP analysis for model interpretability"""
        if self.shap_explainer is None or self.shap_values is None:
            print("SHAP analysis not available")
            return None
        
        print("=== SHAP ANALYSIS ===")
        
        # Create plots directory
        import os
        os.makedirs('../plots/shap', exist_ok=True)
        
        # 1. Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X_test.sample(min(200, len(self.X_test))),
                         feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot')
        if save_plots:
            plt.savefig('../plots/shap/summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature importance plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_test.sample(min(200, len(self.X_test))),
                         plot_type="bar", feature_names=self.feature_names, show=False)
        plt.title('SHAP Feature Importance')
        if save_plots:
            plt.savefig('../plots/shap/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Dependence plots for top features
        if hasattr(self.model, 'feature_importances_'):
            top_features_idx = np.argsort(self.model.feature_importances_)[-5:][::-1]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.ravel()
            
            for i, feature_idx in enumerate(top_features_idx):
                if i < 5:
                    shap.dependence_plot(feature_idx, self.shap_values.values, 
                                       self.X_test.sample(min(200, len(self.X_test))),
                                       feature_names=self.feature_names,
                                       ax=axes[i], show=False)
            
            # Remove empty subplot
            fig.delaxes(axes[5])
            
            plt.suptitle('SHAP Dependence Plots for Top Features', fontsize=16)
            plt.tight_layout()
            if save_plots:
                plt.savefig('../plots/shap/dependence_plots.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("SHAP analysis completed. Plots saved to ../plots/shap/")
    
    def analyze_prediction_segments(self, figsize=(15, 10), save_path=None):
        """Analyze model performance across different prediction segments"""
        print("=== PREDICTION SEGMENTS ANALYSIS ===")
        
        # Create probability bins
        prob_bins = pd.cut(self.y_pred_proba, bins=10, labels=False)
        
        # Analyze each segment
        segment_analysis = []
        
        for segment in range(10):
            mask = prob_bins == segment
            if mask.sum() > 0:
                segment_data = {
                    'Segment': segment + 1,
                    'Probability_Range': f"{segment/10:.1f}-{(segment+1)/10:.1f}",
                    'Count': mask.sum(),
                    'Actual_Churn_Rate': self.y_test[mask].mean(),
                    'Predicted_Churn_Rate': self.y_pred_proba[mask].mean(),
                    'Precision': precision_score(self.y_test[mask], self.y_pred[mask]) if self.y_pred[mask].sum() > 0 else 0,
                    'Recall': recall_score(self.y_test[mask], self.y_pred[mask]) if self.y_test[mask].sum() > 0 else 0
                }
                segment_analysis.append(segment_data)
        
        segment_df = pd.DataFrame(segment_analysis)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Count by segment
        axes[0, 0].bar(segment_df['Segment'], segment_df['Count'])
        axes[0, 0].set_xlabel('Probability Segment')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Sample Distribution by Probability Segment')
        
        # 2. Actual vs Predicted churn rate
        axes[0, 1].plot(segment_df['Segment'], segment_df['Actual_Churn_Rate'], 
                       'o-', label='Actual', color='blue')
        axes[0, 1].plot(segment_df['Segment'], segment_df['Predicted_Churn_Rate'], 
                       'o-', label='Predicted', color='red')
        axes[0, 1].set_xlabel('Probability Segment')
        axes[0, 1].set_ylabel('Churn Rate')
        axes[0, 1].set_title('Actual vs Predicted Churn Rate by Segment')
        axes[0, 1].legend()
        
        # 3. Precision by segment
        axes[1, 0].bar(segment_df['Segment'], segment_df['Precision'])
        axes[1, 0].set_xlabel('Probability Segment')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision by Segment')
        
        # 4. Recall by segment
        axes[1, 1].bar(segment_df['Segment'], segment_df['Recall'])
        axes[1, 1].set_xlabel('Probability Segment')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall by Segment')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nSegment Analysis:")
        print(segment_df.round(4).to_string(index=False))
        
        return segment_df
    
    def business_impact_analysis(self, cost_fp=100, cost_fn=500, revenue_retention=1000):
        """Analyze business impact of model predictions"""
        print("=== BUSINESS IMPACT ANALYSIS ===")
        print(f"Assumptions:")
        print(f"  Cost of False Positive (unnecessary retention effort): ${cost_fp}")
        print(f"  Cost of False Negative (lost customer): ${cost_fn}")
        print(f"  Revenue from successful retention: ${revenue_retention}")
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        
        # Calculate costs and benefits
        cost_false_positives = fp * cost_fp
        cost_false_negatives = fn * cost_fn
        revenue_true_positives = tp * revenue_retention
        
        total_cost = cost_false_positives + cost_false_negatives
        total_benefit = revenue_true_positives
        net_benefit = total_benefit - total_cost
        
        # Calculate baseline (no model) scenario
        baseline_cost = (tp + fn) * cost_fn  # All churners are lost
        
        # Model value
        model_value = baseline_cost - (total_cost - total_benefit)
        
        print(f"\nBusiness Impact Analysis:")
        print(f"  True Positives (Correctly identified churners): {tp}")
        print(f"  False Positives (Incorrectly flagged as churners): {fp}")
        print(f"  False Negatives (Missed churners): {fn}")
        print(f"  True Negatives (Correctly identified non-churners): {tn}")
        
        print(f"\nCosts:")
        print(f"  False Positive Cost: ${cost_false_positives:,.2f}")
        print(f"  False Negative Cost: ${cost_false_negatives:,.2f}")
        print(f"  Total Cost: ${total_cost:,.2f}")
        
        print(f"\nBenefits:")
        print(f"  Revenue from Retention: ${revenue_true_positives:,.2f}")
        
        print(f"\nNet Business Impact:")
        print(f"  Net Benefit: ${net_benefit:,.2f}")
        print(f"  Baseline Cost (no model): ${baseline_cost:,.2f}")
        print(f"  Model Value: ${model_value:,.2f}")
        
        # ROI calculation
        model_roi = (model_value / baseline_cost) * 100
        print(f"  Model ROI: {model_roi:.1f}%")
        
        return {
            'net_benefit': net_benefit,
            'model_value': model_value,
            'baseline_cost': baseline_cost,
            'model_roi': model_roi,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    
    def generate_evaluation_report(self, save_path='../reports/model_evaluation_report.html'):
        """Generate comprehensive HTML evaluation report"""
        print("=== GENERATING EVALUATION REPORT ===")
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Customer Churn Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background-color: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px; }}
                .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <h1 class="header">Customer Churn Model Evaluation Report</h1>
            
            <h2>Model Performance Metrics</h2>
            <div class="metric">
                <h3>Classification Metrics</h3>
                <ul>
                    <li><strong>Accuracy:</strong> {basic_metrics['Accuracy']:.4f}</li>
                    <li><strong>Precision:</strong> {basic_metrics['Precision']:.4f}</li>
                    <li><strong>Recall:</strong> {basic_metrics['Recall']:.4f}</li>
                    <li><strong>F1-Score:</strong> {basic_metrics['F1-Score']:.4f}</li>
                    <li><strong>ROC-AUC:</strong> {basic_metrics['ROC-AUC']:.4f}</li>
                    <li><strong>Average Precision:</strong> {basic_metrics['Average Precision']:.4f}</li>
                </ul>
            </div>
            
            <h2>Model Interpretation</h2>
            <p>This report provides comprehensive evaluation of the customer churn prediction model.</p>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Model shows {'good' if basic_metrics['ROC-AUC'] > 0.8 else 'moderate' if basic_metrics['ROC-AUC'] > 0.7 else 'poor'} performance with ROC-AUC of {basic_metrics['ROC-AUC']:.4f}</li>
                <li>Precision of {basic_metrics['Precision']:.4f} indicates {int(basic_metrics['Precision']*100)}% of predicted churners are actually churners</li>
                <li>Recall of {basic_metrics['Recall']:.4f} means {int(basic_metrics['Recall']*100)}% of actual churners are correctly identified</li>
            </ul>
        </body>
        </html>
        """
        
        # Save report
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Evaluation report saved to: {save_path}")
    
    def run_complete_evaluation(self, save_plots=True):
        """Run complete model evaluation pipeline"""
        print("=== STARTING COMPLETE MODEL EVALUATION ===")
        
        # Create plots directory
        import os
        os.makedirs('../plots', exist_ok=True)
        
        # Basic metrics
        basic_metrics = self.calculate_basic_metrics()
        
        # Visualizations
        self.plot_confusion_matrix(save_path='../plots/confusion_matrix_detailed.png' if save_plots else None)
        self.plot_roc_curve(save_path='../plots/roc_curve_detailed.png' if save_plots else None)
        self.plot_precision_recall_curve(save_path='../plots/precision_recall_curve.png' if save_plots else None)
        self.plot_probability_distribution(save_path='../plots/probability_analysis.png' if save_plots else None)
        
        # Feature importance
        importance_df = self.analyze_feature_importance(save_path='../plots/feature_importance_detailed.png' if save_plots else None)
        
        # SHAP analysis
        self.shap_analysis(save_plots=save_plots)
        
        # Segment analysis
        segment_df = self.analyze_prediction_segments(save_path='../plots/segment_analysis.png' if save_plots else None)
        
        # Business impact
        business_impact = self.business_impact_analysis()
        
        # Generate report
        self.generate_evaluation_report()
        
        print("\n=== COMPLETE EVALUATION COMPLETED ===")
        print("All plots and reports have been generated and saved.")
        
        return {
            'basic_metrics': basic_metrics,
            'feature_importance': importance_df,
            'segment_analysis': segment_df,
            'business_impact': business_impact
        }

if __name__ == "__main__":
    # Example usage
    import joblib
    
    # Load model and data
    model = joblib.load("../models/best_model.joblib")
    X_test = pd.read_csv("../data/X_test_processed.csv")
    y_test = pd.read_csv("../data/y_test.csv").squeeze()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, X_test, y_test)
    
    # Run complete evaluation
    results = evaluator.run_complete_evaluation(save_plots=True)
