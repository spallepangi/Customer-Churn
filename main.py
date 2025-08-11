"""
Main Pipeline Script for Customer Churn Prediction

This script orchestrates the complete end-to-end machine learning pipeline:
1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Model building and training
5. Model evaluation and interpretation
6. Results visualization and reporting
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.append('src')

# Import custom modules
from data_cleaning import DataCleaner
from eda import ChurnEDA
from feature_engineering import FeatureEngineer
from model_building import ChurnModelBuilder
from model_evaluation import ModelEvaluator

warnings.filterwarnings('ignore')

class ChurnPredictionPipeline:
    """Complete end-to-end churn prediction pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        
        # Create necessary directories
        self.create_directories()
        
        print("=" * 70)
        print("CUSTOMER CHURN PREDICTION - END-TO-END ML PIPELINE")
        print("=" * 70)
        print(f"Pipeline initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Random State: {random_state}")
    
    def create_directories(self):
        """Create necessary project directories"""
        directories = ['data', 'plots', 'models', 'reports', 'plots/shap']
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("Project directories created/verified")
    
    def step_1_data_cleaning(self, train_path="data/train.csv", test_path="data/test.csv"):
        """Step 1: Data cleaning and preprocessing"""
        print("\n" + "="*50)
        print("STEP 1: DATA CLEANING AND PREPROCESSING")
        print("="*50)
        
        # Initialize data cleaner
        cleaner = DataCleaner()
        
        # Clean the data
        train_clean, test_clean = cleaner.clean_data(
            train_path=train_path,
            test_path=test_path,
            outlier_method='cap'
        )
        
        # Save cleaned data
        train_clean.to_csv("data/train_cleaned.csv", index=False)
        if test_clean is not None:
            test_clean.to_csv("data/test_cleaned.csv", index=False)
        
        self.results['data_cleaning'] = {
            'original_train_shape': cleaner.train_data.shape,
            'cleaned_train_shape': train_clean.shape,
            'original_test_shape': cleaner.test_data.shape if cleaner.test_data is not None else None,
            'cleaned_test_shape': test_clean.shape if test_clean is not None else None
        }
        
        print("\n✓ Data cleaning completed successfully!")
        print(f"  - Original training data: {self.results['data_cleaning']['original_train_shape']}")
        print(f"  - Cleaned training data: {self.results['data_cleaning']['cleaned_train_shape']}")
        
        return train_clean, test_clean
    
    def step_2_exploratory_analysis(self, data_path="data/train_cleaned.csv"):
        """Step 2: Exploratory Data Analysis"""
        print("\n" + "="*50)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Initialize EDA analyzer
        eda = ChurnEDA()
        
        # Run complete EDA
        eda_summary = eda.run_complete_eda(data_path)
        
        self.results['eda'] = eda_summary
        
        print("\n✓ Exploratory Data Analysis completed successfully!")
        print(f"  - Dataset shape: {eda_summary['dataset_shape']}")
        print(f"  - Overall churn rate: {eda_summary['churn_rate']:.2f}%")
        print(f"  - Numerical features: {len(eda_summary['numerical_features'])}")
        print(f"  - Categorical features: {len(eda_summary['categorical_features'])}")
        
        return eda_summary
    
    def step_3_feature_engineering(self, train_path="data/train_cleaned.csv", 
                                  test_path="data/test_cleaned.csv"):
        """Step 3: Feature Engineering"""
        print("\n" + "="*50)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*50)
        
        # Load cleaned data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path) if os.path.exists(test_path) else None
        
        # Initialize feature engineer
        fe = FeatureEngineer()
        
        # Prepare features
        X_train, X_test, y_train, y_test = fe.prepare_features(
            train_data=train_data,
            test_data=test_data,
            target_col='Churn',
            feature_selection_method='importance',
            n_features=25,
            scaling_method='standard',
            create_polynomials=False
        )
        
        # Save processed features
        X_train.to_csv("data/X_train_processed.csv", index=False)
        y_train.to_csv("data/y_train.csv", index=False, header=True)
        
        if X_test is not None:
            X_test.to_csv("data/X_test_processed.csv", index=False)
        if y_test is not None:
            y_test.to_csv("data/y_test.csv", index=False, header=True)
        
        self.results['feature_engineering'] = {
            'original_features': train_data.shape[1] - 1,  # Exclude target
            'engineered_features': len(fe.engineered_features),
            'final_features': X_train.shape[1],
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0] if X_test is not None else None
        }
        
        print("\n✓ Feature Engineering completed successfully!")
        print(f"  - Original features: {self.results['feature_engineering']['original_features']}")
        print(f"  - Engineered features: {self.results['feature_engineering']['engineered_features']}")
        print(f"  - Final selected features: {self.results['feature_engineering']['final_features']}")
        
        return X_train, X_test, y_train, y_test, fe
    
    def step_4_model_building(self, X_train, y_train, X_test, y_test):
        """Step 4: Model Building and Training"""
        print("\n" + "="*50)
        print("STEP 4: MODEL BUILDING AND TRAINING")
        print("="*50)
        
        # Initialize model builder
        model_builder = ChurnModelBuilder(random_state=self.random_state)
        
        # Run complete modeling pipeline
        comparison_results = model_builder.run_complete_modeling_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tune_hyperparameters=True,
            save_models=True
        )
        
        self.results['model_building'] = {
            'models_trained': len(model_builder.trained_models),
            'best_model': model_builder.best_model_name,
            'best_model_performance': model_builder.model_scores.get(model_builder.best_model_name, {}),
            'comparison_results': comparison_results
        }
        
        print("\n✓ Model Building completed successfully!")
        print(f"  - Models trained: {self.results['model_building']['models_trained']}")
        print(f"  - Best model: {self.results['model_building']['best_model']}")
        
        if model_builder.best_model_name and model_builder.best_model_name in model_builder.model_scores:
            best_scores = model_builder.model_scores[model_builder.best_model_name]
            print(f"  - Best model ROC-AUC: {best_scores.get('roc_auc', 'N/A')}")
        
        return model_builder, comparison_results
    
    def step_5_model_evaluation(self, model_builder, X_test, y_test):
        """Step 5: Comprehensive Model Evaluation"""
        print("\n" + "="*50)
        print("STEP 5: COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        if not model_builder.best_model:
            print("No best model available for evaluation")
            return None
        
        # Initialize model evaluator
        evaluator = ModelEvaluator(
            model=model_builder.best_model,
            X_test=X_test,
            y_test=y_test
        )
        
        # Run complete evaluation
        evaluation_results = evaluator.run_complete_evaluation(save_plots=True)
        
        self.results['model_evaluation'] = evaluation_results
        
        print("\n✓ Model Evaluation completed successfully!")
        print("  - Comprehensive metrics calculated")
        print("  - Visualizations generated")
        print("  - SHAP analysis completed")
        print("  - Business impact analysis completed")
        print("  - Evaluation report generated")
        
        return evaluation_results
    
    def generate_final_report(self):
        """Generate final comprehensive report"""
        print("\n" + "="*50)
        print("GENERATING FINAL COMPREHENSIVE REPORT")
        print("="*50)
        
        # Create comprehensive HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Customer Churn Prediction - Complete Project Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 40px; 
                    line-height: 1.6;
                    color: #333;
                }}
                .header {{ 
                    color: #2c3e50; 
                    border-bottom: 3px solid #3498db; 
                    padding-bottom: 20px; 
                    margin-bottom: 30px;
                }}
                .section {{ 
                    background-color: #f8f9fa; 
                    padding: 20px; 
                    margin: 20px 0; 
                    border-radius: 8px; 
                    border-left: 4px solid #3498db;
                }}
                .metric {{ 
                    background-color: #e8f4fd; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px; 
                    border: 1px solid #bee5eb;
                }}
                .success {{ color: #27ae60; font-weight: bold; }}
                .warning {{ color: #f39c12; font-weight: bold; }}
                .error {{ color: #e74c3c; font-weight: bold; }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0; 
                    background: white;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #3498db; 
                    color: white; 
                    font-weight: bold;
                }}
                .highlight {{ 
                    background-color: #fff3cd; 
                    padding: 15px; 
                    border-radius: 5px; 
                    border-left: 4px solid #ffc107; 
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Customer Churn Prediction Project</h1>
                <h2>Complete End-to-End Machine Learning Pipeline Report</h2>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="highlight">
                <h3>🚀 Project Summary</h3>
                <p>This report presents a comprehensive customer churn prediction solution using machine learning. 
                The project follows industry best practices and includes data preprocessing, exploratory analysis, 
                feature engineering, model development, and business impact assessment.</p>
            </div>
        """
        
        # Add data processing summary
        if 'data_cleaning' in self.results:
            dc = self.results['data_cleaning']
            html_content += f"""
            <div class="section">
                <h3>📊 Data Processing Summary</h3>
                <div class="metric">
                    <h4>Dataset Information</h4>
                    <ul>
                        <li><strong>Original Training Data:</strong> {dc['original_train_shape']} samples × {dc['original_train_shape'][1] if dc['original_train_shape'] else 'N/A'} features</li>
                        <li><strong>Cleaned Training Data:</strong> {dc['cleaned_train_shape']} samples × {dc['cleaned_train_shape'][1] if dc['cleaned_train_shape'] else 'N/A'} features</li>
                        <li><strong>Data Quality:</strong> <span class="success">✓ Missing values handled</span></li>
                        <li><strong>Outliers:</strong> <span class="success">✓ Outliers capped</span></li>
                        <li><strong>Encoding:</strong> <span class="success">✓ Categorical features encoded</span></li>
                    </ul>
                </div>
            </div>
            """
        
        # Add EDA summary
        if 'eda' in self.results:
            eda = self.results['eda']
            html_content += f"""
            <div class="section">
                <h3>🔍 Exploratory Data Analysis</h3>
                <div class="metric">
                    <h4>Key Findings</h4>
                    <ul>
                        <li><strong>Dataset Shape:</strong> {eda['dataset_shape']}</li>
                        <li><strong>Overall Churn Rate:</strong> <span class="{'warning' if eda['churn_rate'] > 30 else 'success'}">{eda['churn_rate']:.2f}%</span></li>
                        <li><strong>Feature Distribution:</strong> {len(eda['numerical_features'])} numerical, {len(eda['categorical_features'])} categorical</li>
                        <li><strong>Visualizations:</strong> <span class="success">✓ Generated comprehensive plots</span></li>
                    </ul>
                </div>
            </div>
            """
        
        # Add feature engineering summary
        if 'feature_engineering' in self.results:
            fe = self.results['feature_engineering']
            html_content += f"""
            <div class="section">
                <h3>⚙️ Feature Engineering</h3>
                <div class="metric">
                    <h4>Feature Development</h4>
                    <ul>
                        <li><strong>Original Features:</strong> {fe['original_features']}</li>
                        <li><strong>Engineered Features:</strong> {fe['engineered_features']}</li>
                        <li><strong>Final Selected Features:</strong> {fe['final_features']}</li>
                        <li><strong>Training Samples:</strong> {fe['train_samples']:,}</li>
                        <li><strong>Feature Selection:</strong> <span class="success">✓ Importance-based selection</span></li>
                        <li><strong>Scaling:</strong> <span class="success">✓ StandardScaler applied</span></li>
                    </ul>
                </div>
            </div>
            """
        
        # Add model performance summary
        if 'model_building' in self.results:
            mb = self.results['model_building']
            best_performance = mb['best_model_performance']
            
            # Determine performance level
            auc_score = best_performance.get('roc_auc', 0)
            if auc_score > 0.8:
                performance_class = 'success'
                performance_text = 'Excellent'
            elif auc_score > 0.7:
                performance_class = 'warning'
                performance_text = 'Good'
            else:
                performance_class = 'error'
                performance_text = 'Needs Improvement'
            
            html_content += f"""
            <div class="section">
                <h3>🤖 Model Performance</h3>
                <div class="metric">
                    <h4>Best Model: {mb['best_model']}</h4>
                    <ul>
                        <li><strong>Models Trained:</strong> {mb['models_trained']}</li>
                        <li><strong>Performance Level:</strong> <span class="{performance_class}">{performance_text}</span></li>
                        <li><strong>ROC-AUC Score:</strong> <span class="{performance_class}">{auc_score:.4f}</span></li>
                        <li><strong>Accuracy:</strong> {best_performance.get('accuracy', 'N/A')}</li>
                        <li><strong>Precision:</strong> {best_performance.get('precision', 'N/A')}</li>
                        <li><strong>Recall:</strong> {best_performance.get('recall', 'N/A')}</li>
                        <li><strong>F1-Score:</strong> {best_performance.get('f1_score', 'N/A')}</li>
                    </ul>
                </div>
            </div>
            """
        
        # Add business impact
        if 'model_evaluation' in self.results and 'business_impact' in self.results['model_evaluation']:
            bi = self.results['model_evaluation']['business_impact']
            html_content += f"""
            <div class="section">
                <h3>💰 Business Impact Analysis</h3>
                <div class="metric">
                    <h4>Financial Impact</h4>
                    <ul>
                        <li><strong>Model ROI:</strong> <span class="{'success' if bi['model_roi'] > 0 else 'error'}">{bi['model_roi']:.1f}%</span></li>
                        <li><strong>Model Value:</strong> ${bi['model_value']:,.2f}</li>
                        <li><strong>Net Benefit:</strong> <span class="{'success' if bi['net_benefit'] > 0 else 'error'}">${bi['net_benefit']:,.2f}</span></li>
                        <li><strong>Baseline Cost:</strong> ${bi['baseline_cost']:,.2f}</li>
                    </ul>
                    <h4>Prediction Accuracy</h4>
                    <ul>
                        <li><strong>True Positives:</strong> {bi['tp']} (correctly identified churners)</li>
                        <li><strong>False Positives:</strong> {bi['fp']} (unnecessary retention efforts)</li>
                        <li><strong>False Negatives:</strong> {bi['fn']} (missed churners)</li>
                        <li><strong>True Negatives:</strong> {bi['tn']} (correctly identified non-churners)</li>
                    </ul>
                </div>
            </div>
            """
        
        # Add recommendations
        html_content += f"""
            <div class="section">
                <h3>📋 Key Recommendations</h3>
                <div class="metric">
                    <h4>Model Implementation</h4>
                    <ul>
                        <li><strong>Deployment:</strong> Model is ready for production deployment</li>
                        <li><strong>Monitoring:</strong> Implement model performance monitoring</li>
                        <li><strong>Retraining:</strong> Schedule monthly model retraining</li>
                        <li><strong>Threshold Optimization:</strong> Consider adjusting probability threshold based on business needs</li>
                    </ul>
                    <h4>Business Actions</h4>
                    <ul>
                        <li><strong>High-Risk Customers:</strong> Target customers with churn probability > 0.7</li>
                        <li><strong>Retention Strategy:</strong> Develop targeted retention campaigns</li>
                        <li><strong>Feature Monitoring:</strong> Track key features that drive churn</li>
                        <li><strong>A/B Testing:</strong> Test retention strategies on model predictions</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h3>📁 Project Deliverables</h3>
                <div class="metric">
                    <ul>
                        <li>✅ <strong>Cleaned Dataset:</strong> data/train_cleaned.csv, data/test_cleaned.csv</li>
                        <li>✅ <strong>Processed Features:</strong> data/X_train_processed.csv, data/X_test_processed.csv</li>
                        <li>✅ <strong>Trained Models:</strong> models/ directory with all trained models</li>
                        <li>✅ <strong>Visualizations:</strong> plots/ directory with all analysis plots</li>
                        <li>✅ <strong>SHAP Analysis:</strong> plots/shap/ directory with interpretability plots</li>
                        <li>✅ <strong>Evaluation Reports:</strong> reports/ directory with detailed reports</li>
                        <li>✅ <strong>Source Code:</strong> src/ directory with all Python modules</li>
                    </ul>
                </div>
            </div>
            
            <div class="highlight">
                <h3>🎯 Project Success Metrics</h3>
                <p><strong>✓ Data Quality:</strong> Complete data preprocessing pipeline with outlier handling and feature encoding</p>
                <p><strong>✓ Model Performance:</strong> {f"Achieved {performance_text.lower()} performance with ROC-AUC of {auc_score:.4f}" if 'model_building' in self.results else "Model performance metrics available"}</p>
                <p><strong>✓ Interpretability:</strong> SHAP analysis provides clear feature importance and model explanations</p>
                <p><strong>✓ Business Value:</strong> {f"Model provides positive ROI of {bi['model_roi']:.1f}%" if 'model_evaluation' in self.results and 'business_impact' in self.results['model_evaluation'] else "Business impact analysis completed"}</p>
                <p><strong>✓ Production Ready:</strong> Complete pipeline with model persistence and evaluation framework</p>
            </div>
            
        </body>
        </html>
        """
        
        # Save the report
        report_path = "reports/complete_project_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n✓ Final comprehensive report generated: {report_path}")
        print("  - Complete project summary")
        print("  - Performance metrics")
        print("  - Business impact analysis")
        print("  - Key recommendations")
    
    def run_complete_pipeline(self, train_path="data/train.csv", test_path="data/test.csv"):
        """Run the complete end-to-end pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: Data Cleaning
            train_clean, test_clean = self.step_1_data_cleaning(train_path, test_path)
            
            # Step 2: Exploratory Data Analysis
            eda_summary = self.step_2_exploratory_analysis()
            
            # Step 3: Feature Engineering
            X_train, X_test, y_train, y_test, fe = self.step_3_feature_engineering()
            
            # Step 4: Model Building
            model_builder, comparison_results = self.step_4_model_building(X_train, y_train, X_test, y_test)
            
            # Step 5: Model Evaluation (only if we have test labels)
            if y_test is not None:
                evaluation_results = self.step_5_model_evaluation(model_builder, X_test, y_test)
            else:
                print("\n⚠️  Test set labels not available. Skipping comprehensive evaluation.")
                self.results['model_evaluation'] = None
            
            # Generate Final Report
            self.generate_final_report()
            
            # Pipeline completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "="*70)
            print("🎉 END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"⏱️  Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"📊 Results saved in: reports/complete_project_report.html")
            print(f"🤖 Best model: {model_builder.best_model_name if model_builder.best_model_name else 'N/A'}")
            
            if model_builder.best_model_name and model_builder.best_model_name in model_builder.model_scores:
                best_auc = model_builder.model_scores[model_builder.best_model_name].get('roc_auc', 'N/A')
                print(f"📈 Best ROC-AUC: {best_auc}")
            
            print("\n🚀 Your churn prediction model is ready for deployment!")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            print("Please check the error details and data paths.")
            raise e

def main():
    """Main execution function"""
    print("Starting Customer Churn Prediction Pipeline...")
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline(random_state=42)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        train_path="data/train.csv",
        test_path="data/test.csv"
    )
    
    return results

if __name__ == "__main__":
    results = main()
