"""
Feature Engineering Module for Customer Churn Prediction

This module handles:
1. Creating new meaningful features
2. Feature transformations
3. Feature selection
4. Feature scaling and normalization
5. Polynomial features and interactions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Class to handle all feature engineering operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.selector = None
        self.feature_names = None
        self.engineered_features = []
        
    def create_engagement_features(self, data):
        """Create engagement-related features"""
        print("\n=== CREATING ENGAGEMENT FEATURES ===")
        
        data_fe = data.copy()
        
        # Engagement Score (normalized composite score)
        engagement_cols = ['ViewingHoursPerWeek', 'AverageViewingDuration', 
                          'ContentDownloadsPerMonth', 'UserRating', 'WatchlistSize']
        
        available_cols = [col for col in engagement_cols if col in data.columns]
        
        if len(available_cols) >= 3:
            # Normalize each component to 0-1 scale
            scaler = MinMaxScaler()
            engagement_normalized = pd.DataFrame(
                scaler.fit_transform(data[available_cols]),
                columns=available_cols,
                index=data.index
            )
            
            data_fe['EngagementScore'] = engagement_normalized.mean(axis=1)
            
            # Engagement categories
            data_fe['EngagementCategory'] = pd.cut(
                data_fe['EngagementScore'], 
                bins=3, 
                labels=[0, 1, 2]  # Low, Medium, High
            ).astype(int)
            
            self.engineered_features.extend(['EngagementScore', 'EngagementCategory'])
            print("Created engagement features: EngagementScore, EngagementCategory")
        
        # Viewing intensity features
        if 'ViewingHoursPerWeek' in data.columns and 'AverageViewingDuration' in data.columns:
            data_fe['ViewingIntensity'] = (data['ViewingHoursPerWeek'] * 60) / data['AverageViewingDuration']
            data_fe['ViewingIntensity'] = data_fe['ViewingIntensity'].fillna(0)
            self.engineered_features.append('ViewingIntensity')
            print("Created ViewingIntensity feature")
        
        # Content consumption rate
        if 'ContentDownloadsPerMonth' in data.columns and 'ViewingHoursPerWeek' in data.columns:
            data_fe['ContentConsumptionRate'] = data['ContentDownloadsPerMonth'] / (data['ViewingHoursPerWeek'] + 1)
            self.engineered_features.append('ContentConsumptionRate')
            print("Created ContentConsumptionRate feature")
        
        return data_fe
    
    def create_financial_features(self, data):
        """Create financial and subscription-related features"""
        print("\n=== CREATING FINANCIAL FEATURES ===")
        
        data_fe = data.copy()
        
        # Average monthly spending
        if 'TotalCharges' in data.columns and 'AccountAge' in data.columns:
            data_fe['AvgMonthlySpend'] = data['TotalCharges'] / (data['AccountAge'] + 1)
            self.engineered_features.append('AvgMonthlySpend')
        
        # Spending efficiency (charges per viewing hour)
        if 'MonthlyCharges' in data.columns and 'ViewingHoursPerWeek' in data.columns:
            data_fe['SpendingEfficiency'] = data['MonthlyCharges'] / ((data['ViewingHoursPerWeek'] * 4.33) + 1)
            self.engineered_features.append('SpendingEfficiency')
        
        # Value perception (UserRating / MonthlyCharges)
        if 'UserRating' in data.columns and 'MonthlyCharges' in data.columns:
            data_fe['ValuePerception'] = data['UserRating'] / (data['MonthlyCharges'] + 1)
            self.engineered_features.append('ValuePerception')
        
        # Subscription type premium indicator
        if 'SubscriptionType' in data.columns:
            # Assuming subscription types are encoded as 0,1,2 for Basic, Standard, Premium
            data_fe['IsPremium'] = (data['SubscriptionType'] == 2).astype(int)
            self.engineered_features.append('IsPremium')
        
        # Price sensitivity groups
        if 'MonthlyCharges' in data.columns:
            data_fe['PriceSegment'] = pd.cut(
                data['MonthlyCharges'], 
                bins=3, 
                labels=[0, 1, 2]  # Low, Medium, High
            ).astype(int)
            self.engineered_features.append('PriceSegment')
        
        print(f"Created financial features: {[f for f in self.engineered_features if 'Spend' in f or 'Price' in f or 'Value' in f or 'Premium' in f]}")
        return data_fe
    
    def create_behavioral_features(self, data):
        """Create behavioral and usage pattern features"""
        print("\n=== CREATING BEHAVIORAL FEATURES ===")
        
        data_fe = data.copy()
        
        # Account tenure categories
        if 'AccountAge' in data.columns:
            data_fe['TenureCategory'] = pd.cut(
                data['AccountAge'], 
                bins=[0, 12, 36, 120], 
                labels=[0, 1, 2]  # New, Established, Veteran
            ).astype(int)
            self.engineered_features.append('TenureCategory')
        
        # Support interaction frequency
        if 'SupportTicketsPerMonth' in data.columns:
            data_fe['HighSupport'] = (data['SupportTicketsPerMonth'] > data['SupportTicketsPerMonth'].median()).astype(int)
            self.engineered_features.append('HighSupport')
        
        # Digital adoption score
        digital_features = ['PaperlessBilling', 'MultiDeviceAccess', 'SubtitlesEnabled']
        available_digital = [col for col in digital_features if col in data.columns]
        
        if len(available_digital) >= 2:
            data_fe['DigitalAdoption'] = data[available_digital].sum(axis=1)
            self.engineered_features.append('DigitalAdoption')
        
        # Content diversity (assuming GenrePreference is encoded)
        if 'GenrePreference' in data.columns and 'ContentType' in data.columns:
            data_fe['ContentDiversity'] = (data['GenrePreference'] != data['ContentType']).astype(int)
            self.engineered_features.append('ContentDiversity')
        
        # Heavy user indicator
        if 'ViewingHoursPerWeek' in data.columns:
            heavy_threshold = data['ViewingHoursPerWeek'].quantile(0.75)
            data_fe['HeavyUser'] = (data['ViewingHoursPerWeek'] > heavy_threshold).astype(int)
            self.engineered_features.append('HeavyUser')
        
        print(f"Created behavioral features: {[f for f in self.engineered_features if 'Category' in f or 'Support' in f or 'Digital' in f or 'Heavy' in f or 'Diversity' in f]}")
        return data_fe
    
    def create_interaction_features(self, data):
        """Create interaction features between important variables"""
        print("\n=== CREATING INTERACTION FEATURES ===")
        
        data_fe = data.copy()
        
        # Account Age × Monthly Charges
        if 'AccountAge' in data.columns and 'MonthlyCharges' in data.columns:
            data_fe['Age_Charges_Interaction'] = data['AccountAge'] * data['MonthlyCharges']
            self.engineered_features.append('Age_Charges_Interaction')
        
        # User Rating × Viewing Hours
        if 'UserRating' in data.columns and 'ViewingHoursPerWeek' in data.columns:
            data_fe['Rating_ViewingHours_Interaction'] = data['UserRating'] * data['ViewingHoursPerWeek']
            self.engineered_features.append('Rating_ViewingHours_Interaction')
        
        # Support Tickets × Account Age
        if 'SupportTicketsPerMonth' in data.columns and 'AccountAge' in data.columns:
            data_fe['Support_Age_Interaction'] = data['SupportTicketsPerMonth'] * data['AccountAge']
            self.engineered_features.append('Support_Age_Interaction')
        
        # Monthly Charges × Viewing Hours (value for money)
        if 'MonthlyCharges' in data.columns and 'ViewingHoursPerWeek' in data.columns:
            data_fe['Charges_ViewingHours_Ratio'] = data['MonthlyCharges'] / (data['ViewingHoursPerWeek'] + 1)
            self.engineered_features.append('Charges_ViewingHours_Ratio')
        
        print(f"Created interaction features: {[f for f in self.engineered_features if 'Interaction' in f or 'Ratio' in f]}")
        return data_fe
    
    def create_statistical_features(self, data):
        """Create statistical aggregation features"""
        print("\n=== CREATING STATISTICAL FEATURES ===")
        
        data_fe = data.copy()
        
        # Customer activity score (composite of multiple engagement metrics)
        activity_cols = ['ViewingHoursPerWeek', 'ContentDownloadsPerMonth', 'WatchlistSize']
        available_activity = [col for col in activity_cols if col in data.columns]
        
        if len(available_activity) >= 2:
            # Z-score normalization for activity metrics
            activity_data = data[available_activity]
            activity_zscore = (activity_data - activity_data.mean()) / activity_data.std()
            data_fe['ActivityScore'] = activity_zscore.mean(axis=1)
            self.engineered_features.append('ActivityScore')
        
        # Risk score based on negative indicators
        risk_indicators = []
        
        if 'UserRating' in data.columns:
            # Low rating is a risk factor
            data_fe['LowRatingRisk'] = (data['UserRating'] < data['UserRating'].quantile(0.3)).astype(int)
            risk_indicators.append('LowRatingRisk')
        
        if 'SupportTicketsPerMonth' in data.columns:
            # High support tickets is a risk factor
            data_fe['HighSupportRisk'] = (data['SupportTicketsPerMonth'] > data['SupportTicketsPerMonth'].quantile(0.7)).astype(int)
            risk_indicators.append('HighSupportRisk')
        
        if 'ViewingHoursPerWeek' in data.columns:
            # Low viewing is a risk factor
            data_fe['LowViewingRisk'] = (data['ViewingHoursPerWeek'] < data['ViewingHoursPerWeek'].quantile(0.3)).astype(int)
            risk_indicators.append('LowViewingRisk')
        
        if len(risk_indicators) >= 2:
            data_fe['RiskScore'] = data_fe[risk_indicators].sum(axis=1)
            self.engineered_features.append('RiskScore')
            self.engineered_features.extend(risk_indicators)
        
        print(f"Created statistical features: ActivityScore, RiskScore, and risk indicators")
        return data_fe
    
    def engineer_features(self, data):
        """Apply all feature engineering techniques"""
        print("=== STARTING FEATURE ENGINEERING ===")
        
        data_fe = data.copy()
        
        # Apply all feature creation methods
        data_fe = self.create_engagement_features(data_fe)
        data_fe = self.create_financial_features(data_fe)
        data_fe = self.create_behavioral_features(data_fe)
        data_fe = self.create_interaction_features(data_fe)
        data_fe = self.create_statistical_features(data_fe)
        
        print(f"\nTotal engineered features created: {len(self.engineered_features)}")
        print(f"Engineered features: {self.engineered_features}")
        
        print(f"\nDataset shape after feature engineering: {data_fe.shape}")
        return data_fe
    
    def select_features(self, X, y, method='correlation', k=20):
        """Select best features using various methods"""
        print(f"\n=== FEATURE SELECTION ({method.upper()}) ===")
        print(f"Original features: {X.shape[1]}")
        
        if method == 'correlation':
            # Remove features with high correlation to target or other features
            corr_matrix = X.corr().abs()
            
            # Features correlated with target
            if 'Churn' in X.columns:
                target_corr = corr_matrix['Churn'].sort_values(ascending=False)
                selected_features = target_corr.head(k).index.tolist()
                if 'Churn' in selected_features:
                    selected_features.remove('Churn')
            else:
                # If target not in X, use univariate selection
                selector = SelectKBest(score_func=f_classif, k=k)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                
        elif method == 'univariate':
            # Univariate statistical tests
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.selector = selector
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.selector = selector
            
        elif method == 'importance':
            # Feature importance from Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = feature_importance.head(k)['feature'].tolist()
            
        else:
            selected_features = X.columns.tolist()
        
        print(f"Selected features: {len(selected_features)}")
        print(f"Top 10 selected features: {selected_features[:10]}")
        
        return selected_features
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        """Scale features using specified method"""
        print(f"\n=== FEATURE SCALING ({method.upper()}) ===")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            print("No scaling applied")
            return X_train, X_test
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_scaled = None
        
        self.scaler = scaler
        print("Feature scaling completed")
        
        return X_train_scaled, X_test_scaled
    
    def create_polynomial_features(self, X, degree=2, interaction_only=True, include_bias=False):
        """Create polynomial and interaction features"""
        print(f"\n=== CREATING POLYNOMIAL FEATURES (degree={degree}) ===")
        
        # Select only numerical features for polynomial expansion
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_numerical = X[numerical_cols]
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        X_poly = poly.fit_transform(X_numerical)
        feature_names = poly.get_feature_names_out(X_numerical.columns)
        
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        # Combine with original categorical features
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            X_final = pd.concat([X_poly_df, X[categorical_cols]], axis=1)
        else:
            X_final = X_poly_df
        
        print(f"Original features: {X.shape[1]}")
        print(f"Features after polynomial expansion: {X_final.shape[1]}")
        
        return X_final
    
    def prepare_features(self, train_data, test_data=None, target_col='Churn',
                        feature_selection_method='importance', n_features=20,
                        scaling_method='standard', create_polynomials=False):
        """Complete feature preparation pipeline"""
        print("=== STARTING FEATURE PREPARATION PIPELINE ===")
        
        # Apply feature engineering
        train_fe = self.engineer_features(train_data)
        
        if test_data is not None:
            test_fe = self.engineer_features(test_data)
        else:
            test_fe = None
        
        # Separate features and target
        X_train = train_fe.drop([target_col, 'CustomerID'], axis=1, errors='ignore')
        y_train = train_fe[target_col]
        
        if test_fe is not None:
            X_test = test_fe.drop([target_col, 'CustomerID'], axis=1, errors='ignore')
            if target_col in test_fe.columns:
                y_test = test_fe[target_col]
            else:
                y_test = None
        else:
            X_test, y_test = None, None
        
        # Create polynomial features if requested
        if create_polynomials:
            X_train = self.create_polynomial_features(X_train)
            if X_test is not None:
                # Apply same transformation to test set
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                numerical_cols = X_test.select_dtypes(include=[np.number]).columns
                X_test_numerical = X_test[numerical_cols]
                X_test_poly = poly.fit_transform(X_test_numerical)
                feature_names = poly.get_feature_names_out(X_test_numerical.columns)
                X_test_poly_df = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)
                
                categorical_cols = X_test.select_dtypes(exclude=[np.number]).columns
                if len(categorical_cols) > 0:
                    X_test = pd.concat([X_test_poly_df, X_test[categorical_cols]], axis=1)
                else:
                    X_test = X_test_poly_df
        
        # Feature selection
        if feature_selection_method != 'none':
            selected_features = self.select_features(X_train, y_train, 
                                                   method=feature_selection_method, k=n_features)
            X_train = X_train[selected_features]
            if X_test is not None:
                # Ensure test set has same features
                available_features = [f for f in selected_features if f in X_test.columns]
                X_test = X_test[available_features]
        
        # Feature scaling
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, method=scaling_method)
        
        print("\n=== FEATURE PREPARATION COMPLETED ===")
        print(f"Final training set shape: {X_train_scaled.shape}")
        if X_test_scaled is not None:
            print(f"Final test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Example usage
    fe = FeatureEngineer()
    
    # Load data (assuming cleaned data exists)
    train_data = pd.read_csv("../data/train_cleaned.csv")
    test_data = pd.read_csv("../data/test_cleaned.csv")
    
    # Prepare features
    X_train, X_test, y_train, y_test = fe.prepare_features(
        train_data, test_data,
        feature_selection_method='importance',
        n_features=25,
        scaling_method='standard'
    )
    
    # Save processed features
    X_train.to_csv("../data/X_train_processed.csv", index=False)
    X_test.to_csv("../data/X_test_processed.csv", index=False)
    y_train.to_csv("../data/y_train.csv", index=False)
    
    print("Feature engineering completed and saved!")
