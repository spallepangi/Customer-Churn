"""
Data Cleaning Module for Customer Churn Prediction

This module handles:
1. Data loading and initial inspection
2. Missing value treatment
3. Data type corrections
4. Outlier detection and handling
5. Data standardization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Class to handle all data cleaning operations"""
    
    def __init__(self):
        self.label_encoders = {}
        self.outlier_bounds = {}
        
    def load_data(self, train_path, test_path=None):
        """Load training and test datasets"""
        print("Loading data...")
        self.train_data = pd.read_csv(train_path)
        print(f"Training data shape: {self.train_data.shape}")
        
        if test_path:
            self.test_data = pd.read_csv(test_path)
            print(f"Test data shape: {self.test_data.shape}")
        else:
            self.test_data = None
            
        return self.train_data, self.test_data
    
    def initial_data_inspection(self, data):
        """Perform initial data inspection"""
        print("\n=== DATA INSPECTION ===")
        print(f"Dataset shape: {data.shape}")
        print(f"\nData types:")
        print(data.dtypes)
        print(f"\nMissing values:")
        print(data.isnull().sum()[data.isnull().sum() > 0])
        print(f"\nBasic statistics:")
        print(data.describe())
        
        return data.info()
    
    def handle_missing_values(self, data, strategy='median'):
        """Handle missing values in the dataset"""
        print("\n=== HANDLING MISSING VALUES ===")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found!")
            return data
        
        print(f"Missing values found:")
        print(missing_values[missing_values > 0])
        
        # Handle numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                if strategy == 'median':
                    data[col].fillna(data[col].median(), inplace=True)
                elif strategy == 'mean':
                    data[col].fillna(data[col].mean(), inplace=True)
                elif strategy == 'mode':
                    data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Handle categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        print("Missing values handled!")
        return data
    
    def correct_data_types(self, data):
        """Correct data types for specific columns"""
        print("\n=== CORRECTING DATA TYPES ===")
        
        # Convert TotalCharges to numeric (handle any string values)
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        
        # Ensure binary columns are properly encoded
        binary_cols = ['PaperlessBilling', 'MultiDeviceAccess', 'ParentalControl', 'SubtitlesEnabled']
        for col in binary_cols:
            if col in data.columns:
                data[col] = data[col].map({'Yes': 1, 'No': 0})
        
        print("Data types corrected!")
        return data
    
    def detect_outliers(self, data, columns=None, method='IQR'):
        """Detect outliers in numerical columns"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        outlier_indices = set()
        
        for col in columns:
            if col in ['CustomerID', 'Churn']:  # Skip ID and target
                continue
                
            if method == 'IQR':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
                outlier_indices.update(outliers)
                
                print(f"{col}: {len(outliers)} outliers detected")
        
        return list(outlier_indices)
    
    def handle_outliers(self, data, method='cap', outlier_indices=None):
        """Handle outliers using specified method"""
        print("\n=== HANDLING OUTLIERS ===")
        
        if outlier_indices is None:
            outlier_indices = self.detect_outliers(data)
        
        if method == 'remove':
            data_clean = data.drop(outlier_indices)
            print(f"Removed {len(outlier_indices)} outlier rows")
        elif method == 'cap':
            data_clean = data.copy()
            for col, (lower_bound, upper_bound) in self.outlier_bounds.items():
                data_clean[col] = np.clip(data_clean[col], lower_bound, upper_bound)
            print("Outliers capped to bounds")
        else:
            data_clean = data.copy()
            print("No outlier handling applied")
        
        return data_clean
    
    def encode_categorical_features(self, data, target_col='Churn'):
        """Encode categorical features"""
        print("\n=== ENCODING CATEGORICAL FEATURES ===")
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['CustomerID']]
        
        data_encoded = data.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data_encoded[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                # For test data, use already fitted encoder
                try:
                    data_encoded[col] = self.label_encoders[col].transform(data[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    data_encoded[col] = self.label_encoders[col].transform(
                        data[col].astype(str).apply(lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0])
                    )
        
        print(f"Encoded {len(categorical_cols)} categorical columns")
        return data_encoded
    
    def clean_data(self, train_path, test_path=None, outlier_method='cap'):
        """Complete data cleaning pipeline"""
        print("=== STARTING DATA CLEANING PIPELINE ===")
        
        # Load data
        train_data, test_data = self.load_data(train_path, test_path)
        
        # Initial inspection
        print("\n--- TRAINING DATA INSPECTION ---")
        self.initial_data_inspection(train_data)
        
        # Clean training data
        train_clean = self.handle_missing_values(train_data.copy())
        train_clean = self.correct_data_types(train_clean)
        train_clean = self.handle_outliers(train_clean, method=outlier_method)
        train_clean = self.encode_categorical_features(train_clean)
        
        # Clean test data if available
        if test_data is not None:
            print("\n--- TEST DATA INSPECTION ---")
            self.initial_data_inspection(test_data)
            
            test_clean = self.handle_missing_values(test_data.copy())
            test_clean = self.correct_data_types(test_clean)
            test_clean = self.encode_categorical_features(test_clean)  # Use fitted encoders
        else:
            test_clean = None
        
        print("\n=== DATA CLEANING COMPLETED ===")
        return train_clean, test_clean

if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner()
    train_clean, test_clean = cleaner.clean_data(
        train_path="../data/train.csv",
        test_path="../data/test.csv"
    )
    
    # Save cleaned data
    train_clean.to_csv("../data/train_cleaned.csv", index=False)
    if test_clean is not None:
        test_clean.to_csv("../data/test_cleaned.csv", index=False)
    
    print("Cleaned data saved successfully!")
