"""
Exploratory Data Analysis Module for Customer Churn Prediction

This module provides:
1. Comprehensive data exploration
2. Churn distribution analysis
3. Feature correlation analysis
4. Visualization of key patterns
5. Statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ChurnEDA:
    """Class for comprehensive EDA of customer churn data"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self, data_path):
        """Load the dataset"""
        self.data = pd.read_csv(data_path)
        print(f"Dataset loaded with shape: {self.data.shape}")
        return self.data
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("=== DATASET OVERVIEW ===")
        print(f"Shape: {self.data.shape}")
        print(f"\nColumn Names:")
        print(self.data.columns.tolist())
        print(f"\nData Types:")
        print(self.data.dtypes)
        print(f"\nMissing Values:")
        print(self.data.isnull().sum())
        
    def churn_distribution(self):
        """Analyze and visualize churn distribution"""
        print("\n=== CHURN DISTRIBUTION ANALYSIS ===")
        
        # Calculate churn statistics
        churn_counts = self.data['Churn'].value_counts()
        churn_pct = self.data['Churn'].value_counts(normalize=True) * 100
        
        print(f"Churn Distribution:")
        print(f"Not Churned (0): {churn_counts[0]} ({churn_pct[0]:.1f}%)")
        print(f"Churned (1): {churn_counts[1]} ({churn_pct[1]:.1f}%)")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        sns.countplot(data=self.data, x='Churn', ax=ax1)
        ax1.set_title('Churn Distribution (Count)')
        ax1.set_xlabel('Churn (0: No, 1: Yes)')
        ax1.set_ylabel('Count')
        
        # Pie chart
        ax2.pie(churn_counts.values, labels=['Not Churned', 'Churned'], 
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Churn Distribution (Percentage)')
        
        plt.tight_layout()
        plt.savefig('../plots/churn_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return churn_counts, churn_pct
    
    def numerical_features_analysis(self):
        """Analyze numerical features and their relationship with churn"""
        print("\n=== NUMERICAL FEATURES ANALYSIS ===")
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['CustomerID', 'Churn']]
        
        # Statistical summary by churn
        print("Statistical Summary by Churn Status:")
        for col in numerical_cols[:5]:  # Show first 5 for brevity
            print(f"\n{col}:")
            print(self.data.groupby('Churn')[col].agg(['mean', 'median', 'std']).round(2))
        
        # Create box plots for key numerical features
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                sns.boxplot(data=self.data, x='Churn', y=col, ax=axes[i])
                axes[i].set_title(f'{col} by Churn Status')
                axes[i].set_xlabel('Churn (0: No, 1: Yes)')
        
        # Remove empty subplots
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('../plots/numerical_features_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return numerical_cols
    
    def categorical_features_analysis(self):
        """Analyze categorical features and their relationship with churn"""
        print("\n=== CATEGORICAL FEATURES ANALYSIS ===")
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'CustomerID']
        
        # If data is already encoded, convert back for better visualization
        if len(categorical_cols) == 0:
            print("Data appears to be encoded. Using all non-numerical columns except ID and target.")
            all_cols = self.data.columns.tolist()
            categorical_cols = [col for col in all_cols 
                             if col not in ['CustomerID', 'Churn', 'AccountAge', 'MonthlyCharges', 
                                          'TotalCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration',
                                          'ContentDownloadsPerMonth', 'UserRating', 'SupportTicketsPerMonth',
                                          'WatchlistSize']]
        
        # Analyze churn rates by categorical features
        churn_by_category = {}
        
        for col in categorical_cols[:6]:  # Analyze first 6 categories
            if col in self.data.columns:
                cross_tab = pd.crosstab(self.data[col], self.data['Churn'], normalize='index') * 100
                churn_by_category[col] = cross_tab
                print(f"\nChurn Rate by {col}:")
                print(cross_tab.round(2))
        
        # Create visualization
        n_cols = 2
        n_rows = 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols[:6]):
            if col in self.data.columns and i < len(axes):
                # Create count plot with churn hue
                sns.countplot(data=self.data, x=col, hue='Churn', ax=axes[i])
                axes[i].set_title(f'{col} Distribution by Churn Status')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].legend(['Not Churned', 'Churned'])
        
        plt.tight_layout()
        plt.savefig('../plots/categorical_features_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return churn_by_category
    
    def correlation_analysis(self):
        """Analyze correlations between features and with target"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Select numerical columns for correlation
        numerical_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numerical_data.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(correlation_matrix)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('../plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature correlation with target
        churn_correlation = correlation_matrix['Churn'].drop('Churn').sort_values(key=abs, ascending=False)
        print("\nFeatures most correlated with Churn:")
        print(churn_correlation.head(10))
        
        # Visualize top correlations with churn
        plt.figure(figsize=(10, 8))
        top_corr = churn_correlation.head(10)
        colors = ['red' if x < 0 else 'green' for x in top_corr.values]
        bars = plt.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
        plt.yticks(range(len(top_corr)), top_corr.index)
        plt.xlabel('Correlation with Churn')
        plt.title('Top 10 Features Correlated with Churn')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_corr.values)):
            plt.text(value + 0.01 if value >= 0 else value - 0.01, i, 
                    f'{value:.3f}', va='center', ha='left' if value >= 0 else 'right')
        
        plt.tight_layout()
        plt.savefig('../plots/churn_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix, churn_correlation
    
    def engagement_metrics_analysis(self):
        """Analyze customer engagement metrics"""
        print("\n=== ENGAGEMENT METRICS ANALYSIS ===")
        
        engagement_cols = ['ViewingHoursPerWeek', 'AverageViewingDuration', 
                          'ContentDownloadsPerMonth', 'UserRating', 'WatchlistSize']
        
        # Check if columns exist
        available_engagement_cols = [col for col in engagement_cols if col in self.data.columns]
        
        if len(available_engagement_cols) == 0:
            print("Engagement columns not found in expected format")
            return
        
        # Create engagement score
        if all(col in self.data.columns for col in available_engagement_cols):
            # Normalize engagement metrics to 0-1 scale
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            engagement_data = self.data[available_engagement_cols].copy()
            engagement_normalized = pd.DataFrame(
                scaler.fit_transform(engagement_data),
                columns=engagement_data.columns
            )
            
            # Calculate engagement score
            self.data['EngagementScore'] = engagement_normalized.mean(axis=1)
            
            # Analyze engagement score by churn
            print("Engagement Score by Churn Status:")
            print(self.data.groupby('Churn')['EngagementScore'].agg(['mean', 'median', 'std']).round(3))
            
            # Visualize engagement score distribution
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            sns.histplot(data=self.data, x='EngagementScore', hue='Churn', kde=True, alpha=0.7)
            plt.title('Engagement Score Distribution by Churn')
            
            plt.subplot(2, 2, 2)
            sns.boxplot(data=self.data, x='Churn', y='EngagementScore')
            plt.title('Engagement Score Box Plot by Churn')
            
            plt.subplot(2, 2, 3)
            engagement_churn = self.data.groupby('Churn')['EngagementScore'].mean()
            plt.bar(['Not Churned', 'Churned'], engagement_churn.values, 
                   color=['green', 'red'], alpha=0.7)
            plt.title('Average Engagement Score by Churn')
            plt.ylabel('Average Engagement Score')
            
            plt.subplot(2, 2, 4)
            # Create engagement segments
            self.data['EngagementSegment'] = pd.cut(self.data['EngagementScore'], 
                                                   bins=3, labels=['Low', 'Medium', 'High'])
            segment_churn = self.data.groupby('EngagementSegment')['Churn'].mean() * 100
            plt.bar(segment_churn.index, segment_churn.values, alpha=0.7)
            plt.title('Churn Rate by Engagement Segment')
            plt.ylabel('Churn Rate (%)')
            
            plt.tight_layout()
            plt.savefig('../plots/engagement_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def subscription_analysis(self):
        """Analyze subscription-related patterns"""
        print("\n=== SUBSCRIPTION ANALYSIS ===")
        
        # Account age vs churn
        if 'AccountAge' in self.data.columns:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            sns.histplot(data=self.data, x='AccountAge', hue='Churn', kde=True, alpha=0.7)
            plt.title('Account Age Distribution by Churn')
            
            plt.subplot(2, 3, 2)
            sns.boxplot(data=self.data, x='Churn', y='AccountAge')
            plt.title('Account Age by Churn Status')
            
            # Monthly charges vs churn
            if 'MonthlyCharges' in self.data.columns:
                plt.subplot(2, 3, 3)
                sns.histplot(data=self.data, x='MonthlyCharges', hue='Churn', kde=True, alpha=0.7)
                plt.title('Monthly Charges Distribution by Churn')
                
                plt.subplot(2, 3, 4)
                sns.boxplot(data=self.data, x='Churn', y='MonthlyCharges')
                plt.title('Monthly Charges by Churn Status')
            
            # Total charges vs churn
            if 'TotalCharges' in self.data.columns:
                plt.subplot(2, 3, 5)
                sns.scatterplot(data=self.data, x='AccountAge', y='TotalCharges', hue='Churn', alpha=0.6)
                plt.title('Account Age vs Total Charges by Churn')
            
            # Support tickets analysis
            if 'SupportTicketsPerMonth' in self.data.columns:
                plt.subplot(2, 3, 6)
                sns.boxplot(data=self.data, x='Churn', y='SupportTicketsPerMonth')
                plt.title('Support Tickets by Churn Status')
            
            plt.tight_layout()
            plt.savefig('../plots/subscription_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def statistical_tests(self):
        """Perform statistical tests for feature significance"""
        print("\n=== STATISTICAL SIGNIFICANCE TESTS ===")
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['CustomerID', 'Churn']]
        
        # T-tests for numerical features
        print("T-test results for numerical features (p-values):")
        significant_features = []
        
        for col in numerical_cols:
            churned = self.data[self.data['Churn'] == 1][col]
            not_churned = self.data[self.data['Churn'] == 0][col]
            
            t_stat, p_value = stats.ttest_ind(churned, not_churned, nan_policy='omit')
            print(f"{col}: p-value = {p_value:.6f}")
            
            if p_value < 0.05:
                significant_features.append((col, p_value))
        
        print(f"\nSignificant features (p < 0.05): {len(significant_features)}")
        for feature, p_val in significant_features:
            print(f"  {feature}: {p_val:.6f}")
        
        return significant_features
    
    def create_summary_report(self):
        """Create a comprehensive EDA summary report"""
        print("\n=== EDA SUMMARY REPORT ===")
        
        # Basic dataset info
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Missing Values: {self.data.isnull().sum().sum()}")
        
        # Churn distribution
        churn_rate = self.data['Churn'].mean() * 100
        print(f"Overall Churn Rate: {churn_rate:.2f}%")
        
        # Feature types
        numerical_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical Features: {len(numerical_features)}")
        print(f"Categorical Features: {len(categorical_features)}")
        
        return {
            'dataset_shape': self.data.shape,
            'churn_rate': churn_rate,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }
    
    def run_complete_eda(self, data_path):
        """Run complete EDA pipeline"""
        print("=== STARTING COMPREHENSIVE EDA ===")
        
        # Load data
        self.load_data(data_path)
        
        # Create plots directory
        import os
        os.makedirs('../plots', exist_ok=True)
        
        # Run all analyses
        self.basic_info()
        self.churn_distribution()
        self.numerical_features_analysis()
        self.categorical_features_analysis()
        self.correlation_analysis()
        self.engagement_metrics_analysis()
        self.subscription_analysis()
        self.statistical_tests()
        summary = self.create_summary_report()
        
        print("\n=== EDA COMPLETED ===")
        print("All plots saved in ../plots/ directory")
        
        return summary

if __name__ == "__main__":
    # Example usage
    eda = ChurnEDA()
    summary = eda.run_complete_eda("../data/train.csv")
