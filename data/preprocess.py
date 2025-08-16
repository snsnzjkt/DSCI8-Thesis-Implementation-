import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from config import config

class CICIDSPreprocessor:
    """Simple and efficient CIC-IDS2017 preprocessor"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.isolation_forest = IsolationForest(contamination=0.01, random_state=42)
        
    def load_data(self, file_path=None):
        """Load CIC-IDS2017 dataset"""
        if file_path is None:
            # Try common file locations
            possible_paths = [
                f"{config.DATA_DIR}/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                f"{config.DATA_DIR}/Wednesday-workingHours.pcap_ISCX.csv",
                f"{config.DATA_DIR}/cicids2017_sample.csv"  # If you have a sample
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            
            if file_path is None:
                # Create sample data for testing
                return self._create_sample_data()
        
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        return df
    
    def _create_sample_data(self):
        """Create sample data for testing (remove this when you have real data)"""
        print("⚠️ Creating sample data for testing...")
        print("   Replace this with actual CIC-IDS2017 data loading")
        
        np.random.seed(42)
        n_samples = 10000
        n_features = 78
        
        # Create synthetic network traffic features
        X = np.random.rand(n_samples, n_features)
        
        # Create attack labels (15 attack types + 1 benign)
        attack_types = [
            'BENIGN', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
            'DoS slowloris', 'FTP-Patator', 'SSH-Patator', 'Web Attack – Brute Force',
            'Web Attack – Sql Injection', 'Web Attack – XSS', 'Infiltration',
            'Bot', 'PortScan', 'Heartbleed'
        ]
        
        # Generate labels with realistic distribution (mostly benign)
        labels = np.random.choice(attack_types, n_samples, 
                                p=[0.8] + [0.2/14]*14)  # 80% benign, 20% attacks
        
        # Create DataFrame
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['Label'] = labels
        
        return df
    
    def clean_data(self, df):
        """Clean and prepare the dataset"""
        print("🧹 Cleaning data...")
        
        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))
        
        # Remove constant features
        constant_cols = df.columns[df.nunique() <= 1]
        if len(constant_cols) > 0:
            print(f"   Removing {len(constant_cols)} constant features")
            df = df.drop(columns=constant_cols)
        
        # Separate features and labels
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"   Dataset shape: {X.shape}")
        print(f"   Classes: {len(self.label_encoder.classes_)}")
        print(f"   Class distribution: {np.bincount(y_encoded)}")
        
        return X, y_encoded
    
    def normalize_features(self, X_train, X_test):
        """Z-score normalization"""
        print("📊 Normalizing features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def remove_outliers(self, X, y):
        """Remove outliers using Isolation Forest"""
        print("🔍 Removing outliers...")
        
        outlier_mask = self.isolation_forest.fit_predict(X) == 1
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]
        
        removed = len(X) - len(X_clean)
        print(f"   Removed {removed} outliers ({removed/len(X)*100:.1f}%)")
        
        return X_clean, y_clean
    
    def balance_classes(self, X, y):
        """Balance classes using SMOTE"""
        print("⚖️ Balancing classes...")
        
        original_counts = np.bincount(y)
        print(f"   Original distribution: {original_counts}")
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        balanced_counts = np.bincount(y_balanced)
        print(f"   Balanced distribution: {balanced_counts}")
        
        return X_balanced, y_balanced
    
    def split_data(self, X, y):
        """Temporal split (70% train, 30% test)"""
        print("✂️ Splitting data...")
        
        # For simplicity, use random split
        # In real implementation, use temporal split based on timestamps
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_full_pipeline(self):
        """Complete preprocessing pipeline"""
        print("🚀 Starting CIC-IDS2017 preprocessing pipeline...\n")
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Clean data
        X, y = self.clean_data(df)
        
        # Step 3: Split data first (before normalization)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Step 4: Remove outliers from training data only
        X_train_clean, y_train_clean = self.remove_outliers(X_train, y_train)
        
        # Step 5: Balance training data
        X_train_balanced, y_train_balanced = self.balance_classes(X_train_clean, y_train_clean)
        
        # Step 6: Normalize features
        X_train_final, X_test_final = self.normalize_features(X_train_balanced, X_test)
        
        # Step 7: Save processed data
        self.save_processed_data(
            X_train_final, X_test_final, 
            y_train_balanced, y_test
        )
        
        print("\n✅ Preprocessing complete!")
        return X_train_final, X_test_final, y_train_balanced, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        print("💾 Saving processed data...")
        
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': [f'feature_{i}' for i in range(X_train.shape[1])]
        }
        
        with open(f"{config.DATA_DIR}/processed_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"   Saved to: {config.DATA_DIR}/processed_data.pkl")

def main():
    """Run preprocessing pipeline"""
    preprocessor = CICIDSPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_full_pipeline()
    
    print(f"\n📊 Final dataset statistics:")
    print(f"   Training features: {X_train.shape}")
    print(f"   Test features: {X_test.shape}")
    print(f"   Training labels: {y_train.shape}")
    print(f"   Test labels: {y_test.shape}")
    print(f"   Number of classes: {len(np.unique(y_train))}")

if __name__ == "__main__":
    main()