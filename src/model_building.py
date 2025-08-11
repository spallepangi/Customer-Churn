"""
Model Building Module for Customer Churn Prediction

This module handles:
1. Multiple ML algorithm implementation
2. Hyperparameter tuning
3. Cross-validation
4. Model comparison
5. Model persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnModelBuilder:
    """Class for building and comparing churn prediction models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize different ML models"""
        print("=== INITIALIZING ML MODELS ===")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_estimators=100
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True
            ),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
    def train_models(self, X_train, y_train, cv_folds=5):
        """Train all models with cross-validation"""
        print("\n=== TRAINING MODELS ===")
        
        # Initialize models if not done
        if not self.models:
            self.initialize_models()
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                
                self.model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                print(f"  Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
        
        print("\nModel training completed!")
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        print("\n=== EVALUATING MODELS ON TEST SET ===")
        
        evaluation_results = {}
        
        for name, model in self.trained_models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                evaluation_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # Update model scores with test results
                self.model_scores[name].update(evaluation_results[name])
                
                print(f"  Accuracy:  {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F1-Score:  {f1:.4f}")
                print(f"  ROC-AUC:   {auc:.4f}")
                
            except Exception as e:
                print(f"  Error evaluating {name}: {str(e)}")
        
        return evaluation_results
    
    def hyperparameter_tuning(self, X_train, y_train, models_to_tune=None, cv_folds=5):
        """Perform hyperparameter tuning for selected models"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        if models_to_tune is None:
            models_to_tune = ['Random Forest', 'XGBoost', 'Logistic Regression']
        
        # Define parameter grids
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name in models_to_tune:
            if model_name in self.models and model_name in param_grids:
                print(f"\nTuning {model_name}...")
                
                try:
                    # Grid search with cross-validation
                    grid_search = GridSearchCV(
                        estimator=self.models[model_name],
                        param_grid=param_grids[model_name],
                        cv=cv,
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    # Update the model with best parameters
                    self.trained_models[f'{model_name}_Tuned'] = grid_search.best_estimator_
                    
                    print(f"  Best parameters: {grid_search.best_params_}")
                    print(f"  Best CV score: {grid_search.best_score_:.4f}")
                    
                    # Store tuned model scores
                    self.model_scores[f'{model_name}_Tuned'] = {
                        'cv_mean': grid_search.best_score_,
                        'best_params': grid_search.best_params_
                    }
                    
                except Exception as e:
                    print(f"  Error tuning {model_name}: {str(e)}")
        
        print("\nHyperparameter tuning completed!")
    
    def compare_models(self, metric='roc_auc'):
        """Compare all models and identify the best performer"""
        print(f"\n=== MODEL COMPARISON (by {metric.upper()}) ===")
        
        if not self.model_scores:
            print("No model scores available. Train models first.")
            return
        
        # Create comparison dataframe
        comparison_data = []
        
        for name, scores in self.model_scores.items():
            if metric in scores:
                comparison_data.append({
                    'Model': name,
                    'Cross_Val_Mean': scores.get('cv_mean', 'N/A'),
                    'Accuracy': scores.get('accuracy', 'N/A'),
                    'Precision': scores.get('precision', 'N/A'),
                    'Recall': scores.get('recall', 'N/A'),
                    'F1_Score': scores.get('f1_score', 'N/A'),
                    'ROC_AUC': scores.get('roc_auc', 'N/A')
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by the specified metric
        if metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(metric, ascending=False)
        
        print("\nModel Performance Comparison:")
        print("=" * 80)
        print(comparison_df.round(4).to_string(index=False))
        
        # Identify best model
        if len(comparison_df) > 0:
            self.best_model_name = comparison_df.iloc[0]['Model']
            self.best_model = self.trained_models[self.best_model_name]
            
            print(f"\nBest Model: {self.best_model_name}")
            print(f"Best {metric}: {comparison_df.iloc[0][metric.replace('_', ' ').title()]:.4f}")
        
        return comparison_df
    
    def create_evaluation_plots(self, X_test, y_test, save_plots=True):
        """Create comprehensive evaluation plots"""
        print("\n=== CREATING EVALUATION PLOTS ===")
        
        if not self.trained_models:
            print("No trained models available.")
            return
        
        # Create plots directory
        import os
        os.makedirs('../plots', exist_ok=True)
        
        # 1. Model Comparison Bar Plot
        plt.figure(figsize=(15, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.model_scores.keys())
        
        # Prepare data for plotting
        plot_data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            for metric in metrics:
                score = self.model_scores[model_name].get(metric, 0)
                plot_data[metric].append(score if score != 'N/A' else 0)
        
        # Create subplot for each metric
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i + 1)
            bars = plt.bar(model_names, plot_data[metric])
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(metric.replace("_", " ").title())
            
            # Add value labels on bars
            for bar, value in zip(bars, plot_data[metric]):
                if value > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('../plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. ROC Curves for all models
        plt.figure(figsize=(12, 8))
        
        for name, model in self.trained_models.items():
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
            except:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('../plots/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Confusion Matrix for Best Model
        if self.best_model and self.best_model_name:
            plt.figure(figsize=(8, 6))
            
            y_pred = self.best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Not Churned', 'Churned'],
                       yticklabels=['Not Churned', 'Churned'])
            plt.title(f'Confusion Matrix - {self.best_model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            if save_plots:
                plt.savefig('../plots/best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Classification Report
            print(f"\nClassification Report - {self.best_model_name}:")
            print("=" * 60)
            print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
    
    def feature_importance_analysis(self, X_train, top_n=20):
        """Analyze feature importance for tree-based models"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
        
        plt.figure(figsize=(15, 10))
        subplot_idx = 1
        
        for model_name in tree_models:
            if model_name in self.trained_models:
                model = self.trained_models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    # Get feature importances
                    importances = model.feature_importances_
                    feature_names = X_train.columns
                    
                    # Create DataFrame and sort
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Plot top features
                    plt.subplot(2, 2, subplot_idx)
                    top_features = importance_df.head(top_n)
                    
                    bars = plt.barh(range(len(top_features)), top_features['importance'])
                    plt.yticks(range(len(top_features)), top_features['feature'])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top {top_n} Features - {model_name}')
                    plt.gca().invert_yaxis()
                    
                    # Add value labels
                    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                                f'{importance:.3f}', va='center', ha='left', fontsize=8)
                    
                    subplot_idx += 1
                    
                    # Print top features
                    print(f"\nTop 10 Features - {model_name}:")
                    print(importance_df.head(10).to_string(index=False))
        
        plt.tight_layout()
        plt.savefig('../plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, save_path='../models/'):
        """Save trained models to disk"""
        print("\n=== SAVING MODELS ===")
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for name, model in self.trained_models.items():
            try:
                # Clean filename
                filename = name.replace(' ', '_').replace('-', '_') + '.joblib'
                filepath = os.path.join(save_path, filename)
                
                joblib.dump(model, filepath)
                print(f"Saved {name} to {filepath}")
                
            except Exception as e:
                print(f"Error saving {name}: {str(e)}")
        
        # Save model scores
        scores_path = os.path.join(save_path, 'model_scores.joblib')
        joblib.dump(self.model_scores, scores_path)
        print(f"Saved model scores to {scores_path}")
        
        print("Model saving completed!")
    
    def load_models(self, load_path='../models/'):
        """Load saved models from disk"""
        print(f"\n=== LOADING MODELS FROM {load_path} ===")
        
        import os
        
        if not os.path.exists(load_path):
            print(f"Path {load_path} does not exist.")
            return
        
        # Load model scores
        scores_path = os.path.join(load_path, 'model_scores.joblib')
        if os.path.exists(scores_path):
            self.model_scores = joblib.load(scores_path)
            print("Loaded model scores")
        
        # Load individual models
        for filename in os.listdir(load_path):
            if filename.endswith('.joblib') and filename != 'model_scores.joblib':
                try:
                    model_name = filename.replace('.joblib', '').replace('_', ' ')
                    filepath = os.path.join(load_path, filename)
                    
                    model = joblib.load(filepath)
                    self.trained_models[model_name] = model
                    print(f"Loaded {model_name}")
                    
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        print("Model loading completed!")
        
        # Set best model if available
        if self.model_scores:
            best_auc = 0
            for name, scores in self.model_scores.items():
                if scores.get('roc_auc', 0) > best_auc:
                    best_auc = scores['roc_auc']
                    self.best_model_name = name
                    if name in self.trained_models:
                        self.best_model = self.trained_models[name]
    
    def predict_churn_probability(self, X_new):
        """Predict churn probability for new data using best model"""
        if not self.best_model:
            print("No best model available. Train models first.")
            return None
        
        try:
            probabilities = self.best_model.predict_proba(X_new)[:, 1]
            predictions = self.best_model.predict(X_new)
            
            results = pd.DataFrame({
                'Prediction': predictions,
                'Churn_Probability': probabilities
            })
            
            return results
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def run_complete_modeling_pipeline(self, X_train, y_train, X_test, y_test, 
                                     tune_hyperparameters=True, save_models=True):
        """Run the complete modeling pipeline"""
        print("=== STARTING COMPLETE MODELING PIPELINE ===")
        
        # Initialize and train models
        self.initialize_models()
        self.train_models(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            self.hyperparameter_tuning(X_train, y_train)
            # Re-evaluate tuned models
            tuned_models = {k: v for k, v in self.trained_models.items() if 'Tuned' in k}
            if tuned_models:
                for name, model in tuned_models.items():
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics for tuned models
                    self.model_scores[name].update({
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba)
                    })
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Create evaluation plots
        self.create_evaluation_plots(X_test, y_test)
        
        # Feature importance analysis
        self.feature_importance_analysis(X_train)
        
        # Save models if requested
        if save_models:
            self.save_models()
        
        print("\n=== MODELING PIPELINE COMPLETED ===")
        print(f"Best model: {self.best_model_name}")
        
        return comparison_df

if __name__ == "__main__":
    # Example usage
    model_builder = ChurnModelBuilder(random_state=42)
    
    # Load processed data
    X_train = pd.read_csv("../data/X_train_processed.csv")
    X_test = pd.read_csv("../data/X_test_processed.csv")
    y_train = pd.read_csv("../data/y_train.csv").squeeze()
    y_test = pd.read_csv("../data/y_test.csv").squeeze() if os.path.exists("../data/y_test.csv") else None
    
    # Run complete pipeline
    comparison_results = model_builder.run_complete_modeling_pipeline(
        X_train, y_train, X_test, y_test,
        tune_hyperparameters=True,
        save_models=True
    )
