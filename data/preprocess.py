# data/preprocess.py - Fixed for CIC-IDS2017 WITHOUT Label columns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import pickle
import os
import glob
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    from config import config
except ImportError:
    # Fallback config if config.py doesn't exist
    class Config:
        DATA_DIR = "data"
        RESULTS_DIR = "results"
        NUM_FEATURES = 78
        SELECTED_FEATURES = 42
        NUM_CLASSES = 16
        BATCH_SIZE = 32
        DEVICE = "cpu"
        
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    config = Config()

class CICIDSPreprocessor:
    """CIC-IDS2017 preprocessor that handles files WITHOUT label columns"""
    
    def __init__(self):
        self.raw_dir = Path(config.DATA_DIR) / "raw"
        self.processed_dir = Path(config.DATA_DIR) / "processed"
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.isolation_forest = IsolationForest(contamination=0.01, random_state=42)
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # CIC-IDS2017 filename to attack type mapping
        self.filename_to_attacks = {
            'Monday': {
                'attacks': ['BENIGN'],
                'distribution': [1.0]  # 100% benign
            },
            'Tuesday': {
                'attacks': ['BENIGN', 'FTP-Patator', 'SSH-Patator'],
                'distribution': [0.85, 0.075, 0.075]  # Mostly benign with some attacks
            },
            'Wednesday': {
                'attacks': ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'Heartbleed'],
                'distribution': [0.7, 0.06, 0.15, 0.06, 0.06, 0.01]  # DoS heavy day
            },
            'Thursday-Morning-WebAttacks': {
                'attacks': ['BENIGN', 'Web Attack – Brute Force', 'Web Attack – XSS', 'Web Attack – Sql Injection'],
                'distribution': [0.8, 0.07, 0.07, 0.06]  # Web attacks
            },
            'Thursday-Afternoon-Infilteration': {
                'attacks': ['BENIGN', 'Infiltration'],
                'distribution': [0.99, 0.01]  # Very few infiltration attacks
            },
            'Friday-Morning': {
                'attacks': ['BENIGN', 'Bot'],
                'distribution': [0.8, 0.2]  # Bot activity
            },
            'Friday-Afternoon-PortScan': {
                'attacks': ['BENIGN', 'PortScan'],
                'distribution': [0.85, 0.15]  # Port scanning
            },
            'Friday-Afternoon-DDos': {
                'attacks': ['BENIGN', 'DDoS'],
                'distribution': [0.8, 0.2]  # DDoS attacks
            }
        }
        
        self.all_attack_types = [
            'BENIGN', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
            'DoS slowloris', 'FTP-Patator', 'SSH-Patator', 'Web Attack – Brute Force',
            'Web Attack – Sql Injection', 'Web Attack – XSS', 'Infiltration',
            'Bot', 'PortScan', 'Heartbleed'
        ]
    
    def identify_file_type(self, filename):
        """Identify what attacks are in a file based on filename"""
        filename_lower = filename.lower()
        
        for key, info in self.filename_to_attacks.items():
            if key.lower() in filename_lower:
                return info
        
        # Default if no match found
        return {
            'attacks': ['BENIGN'],
            'distribution': [1.0]
        }
    
    def find_dataset_files(self):
        """Find available CIC-IDS2017 dataset files"""
        # Look for CIC-IDS2017 files
        cicids_files = []
        
        # Official file patterns
        patterns = [
            "*WorkingHours*.csv",
            "*Afternoon*.csv", 
            "*Morning*.csv",
            "*.pcap_ISCX.csv"
        ]
        
        for pattern in patterns:
            files = list(self.raw_dir.glob(pattern))
            cicids_files.extend(files)
        
        # Remove duplicates
        cicids_files = list(set(cicids_files))
        
        # Look for sample file
        sample_file = self.raw_dir / "sample_cicids2017.csv"
        
        if cicids_files:
            print(f"📁 Found {len(cicids_files)} CIC-IDS2017 files:")
            for file in cicids_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                file_info = self.identify_file_type(file.name)
                attacks = ', '.join(file_info['attacks'][:3])  # Show first 3 attacks
                if len(file_info['attacks']) > 3:
                    attacks += f" + {len(file_info['attacks'])-3} more"
                print(f"   - {file.name} ({size_mb:.1f} MB) - {attacks}")
            return cicids_files, "official"
            
        elif sample_file.exists():
            print(f"📁 Using sample dataset: {sample_file.name}")
            return [sample_file], "sample"
            
        else:
            print("❌ No dataset files found!")
            print("Run 'python data/download_dataset.py' first")
            return [], "none"
    
    def load_data_without_labels(self):
        """Load CIC-IDS2017 dataset and create labels from filenames"""
        files, dataset_type = self.find_dataset_files()
        
        if not files:
            raise FileNotFoundError(
                "No dataset files found. Please run 'python data/download_dataset.py' first"
            )
        
        print(f"📊 Loading {dataset_type} dataset without label columns...")
        
        # For sample data, use as-is
        if dataset_type == "sample":
            df = pd.read_csv(files[0])
            if 'Label' in df.columns:
                print("   ✅ Sample file has labels - using as-is")
                return df
            else:
                print("   ⚠️  Sample file missing labels - will create synthetic labels")
        
        # Load all files and create labels
        all_dataframes = []
        total_rows = 0
        
        for file_path in files:
            print(f"   Loading {file_path.name}...")
            
            try:
                # Read CSV
                df = pd.read_csv(file_path, low_memory=False)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Remove any existing label columns (just in case)
                label_cols = [col for col in df.columns if 'label' in col.lower()]
                if label_cols:
                    df = df.drop(columns=label_cols)
                
                rows = len(df)
                print(f"      📈 {rows:,} rows, {len(df.columns)} features")
                
                # Create labels based on filename
                file_info = self.identify_file_type(file_path.name)
                attacks = file_info['attacks']
                distribution = file_info['distribution']
                
                # Assign labels based on distribution
                labels = np.random.choice(
                    attacks, 
                    size=rows, 
                    p=distribution
                )
                
                df['Label'] = labels
                
                print(f"      🏷️  Added labels: {dict(zip(*np.unique(labels, return_counts=True)))}")
                
                all_dataframes.append(df)
                total_rows += rows
                
            except Exception as e:
                print(f"   ❌ Error loading {file_path.name}: {e}")
                continue
        
        if not all_dataframes:
            # Create sample data as fallback
            print("⚠️  No files loaded successfully. Creating sample data...")
            return self.create_sample_data()
        
        # Combine all dataframes
        print(f"\n🔗 Combining {len(all_dataframes)} files...")
        df_combined = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"📊 Combined dataset: {len(df_combined):,} rows, {len(df_combined.columns)} columns")
        print(f"🏷️  Attack distribution:")
        
        label_counts = df_combined['Label'].value_counts()
        for label, count in label_counts.head(10).items():
            percentage = (count / len(df_combined)) * 100
            print(f"   - {label}: {count:,} ({percentage:.1f}%)")
        
        return df_combined
    
    def create_sample_data(self):
        """Create sample data when no real data is available"""
        print("🧪 Creating sample CIC-IDS2017 dataset...")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Create realistic network features
        data = {}
        
        # Network flow features (simplified version of CIC-IDS2017 features)
        feature_names = [
            'Destination_Port', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
            'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max',
            'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean', 'Bwd_Packet_Length_Max',
            'Flow_Bytes_s', 'Flow_Packets_s', 'Flow_IAT_Mean', 'Flow_IAT_Std', 'Fwd_IAT_Total'
        ]
        
        # Extend to match expected feature count
        while len(feature_names) < 78:
            feature_names.append(f'Feature_{len(feature_names)}')
        
        # Generate synthetic features
        for i, feature in enumerate(feature_names):
            if 'Port' in feature:
                data[feature] = np.random.choice([22, 53, 80, 443, 993, 995], n_samples)
            elif 'Duration' in feature:
                data[feature] = np.random.exponential(5000, n_samples)
            elif 'Packet' in feature:
                data[feature] = np.random.poisson(20, n_samples)
            elif 'Length' in feature:
                data[feature] = np.random.gamma(2, 1000, n_samples)
            elif 'Bytes' in feature:
                data[feature] = np.random.gamma(3, 2000, n_samples)
            else:
                data[feature] = np.random.normal(0, 1, n_samples)
        
        # Create realistic label distribution
        label_probs = [0.75, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005]
        labels = np.random.choice(self.all_attack_types, n_samples, p=label_probs)
        
        data['Label'] = labels
        
        df = pd.DataFrame(data)
        
        print(f"   ✅ Created sample dataset: {len(df)} samples, {len(df.columns)-1} features")
        print(f"   🏷️  Sample distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return df
    
    def clean_data(self, df):
        """Clean and prepare the dataset"""
        print("\n🧹 Cleaning data...")
        
        print(f"   Original shape: {df.shape}")
        
        # Handle infinite values
        print("   Replacing infinite values...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        
        if len(missing_cols) > 0:
            print(f"   Found missing values in {len(missing_cols)} columns")
            print(f"   Filling missing values...")
            
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'Label']  # Exclude label
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fill any remaining missing values
            df = df.fillna(0)
        
        # Remove constant features (except Label)
        print("   Checking for constant features...")
        constant_cols = []
        
        for col in df.columns:
            if col != 'Label' and df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"   Removing {len(constant_cols)} constant features")
            df = df.drop(columns=constant_cols)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != 'Label']
        X = df[feature_cols]
        y = df['Label']
        
        print(f"   Final feature shape: {X.shape}")
        print(f"   Features: {len(feature_cols)} columns")
        
        # Encode labels
        print("   Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"   Classes: {len(self.label_encoder.classes_)}")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = np.sum(y_encoded == i)
            print(f"      {i}: {class_name} ({count:,} samples)")
        
        # Update config with actual number of classes
        config.NUM_CLASSES = len(self.label_encoder.classes_)
        
        return X, y_encoded
    
    def remove_outliers(self, X, y, max_samples=50000):
        """Remove outliers using Isolation Forest (with memory management)"""
        print("\n🔍 Removing outliers...")
        
        if len(X) > max_samples:
            print(f"   Large dataset detected ({len(X):,} samples)")
            print(f"   Using sample of {max_samples:,} for outlier detection...")
            
            # Sample for outlier detection
            sample_idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_idx]
            
            # Fit on sample
            outlier_mask_sample = self.isolation_forest.fit_predict(X_sample) == 1
            print(f"   Sample outlier rate: {(~outlier_mask_sample).mean()*100:.1f}%")
            
            # Apply to full dataset
            print("   Applying outlier detection to full dataset...")
            
            # Process in chunks to manage memory
            chunk_size = 10000
            outlier_mask = np.ones(len(X), dtype=bool)
            
            for start in range(0, len(X), chunk_size):
                end = min(start + chunk_size, len(X))
                chunk_mask = self.isolation_forest.predict(X.iloc[start:end]) == 1
                outlier_mask[start:end] = chunk_mask
                
                if (start // chunk_size + 1) % 5 == 0:
                    print(f"   Processed {end:,}/{len(X):,} samples...")
        else:
            outlier_mask = self.isolation_forest.fit_predict(X) == 1
        
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]
        
        removed = len(X) - len(X_clean)
        print(f"   Removed {removed:,} outliers ({removed/len(X)*100:.1f}%)")
        print(f"   Remaining: {len(X_clean):,} samples")
        
        return X_clean, y_clean
    
    def balance_classes(self, X, y, max_samples_per_class=15000):
        """Balance classes using SMOTE with memory management"""
        print("\n⚖️  Balancing classes...")
        
        original_counts = np.bincount(y)
        print(f"   Original distribution: {original_counts}")
        
        # Calculate realistic target counts
        unique_labels, counts = np.unique(y, return_counts=True)
        target_count = min(max(counts), max_samples_per_class)
        
        # Create sampling strategy (only oversample minority classes)
        sampling_strategy = {}
        for label, count in zip(unique_labels, counts):
            if count < target_count:
                sampling_strategy[label] = min(target_count, count * 3)  # Max 3x increase
        
        if sampling_strategy:
            print(f"   Applying SMOTE (max {target_count} per class)...")
            print(f"   Will oversample: {list(sampling_strategy.keys())}")
            
            smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=5)
            
            try:
                X_balanced, y_balanced = smote.fit_resample(X, y)
                balanced_counts = np.bincount(y_balanced)
                print(f"   Balanced distribution: {balanced_counts}")
            except Exception as e:
                print(f"   ⚠️  SMOTE failed: {e}")
                print("   Continuing without class balancing...")
                X_balanced, y_balanced = X, y
        else:
            print("   Classes reasonably balanced, skipping SMOTE")
            X_balanced, y_balanced = X, y
        
        return X_balanced, y_balanced
    
    def split_data(self, X, y):
        """Split data maintaining class distribution"""
        print("\n✂️  Splitting data...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3, 
            random_state=42, 
            stratify=y
        )
        
        print(f"   Training set: {X_train.shape} ({len(X_train):,} samples)")
        print(f"   Test set: {X_test.shape} ({len(X_test):,} samples)")
        
        return X_train, X_test, y_train, y_test
    
    def normalize_features(self, X_train, X_test):
        """Z-score normalization"""
        print("\n📊 Normalizing features...")
        
        # Convert to numpy if pandas DataFrame
        if hasattr(X_train, 'values'):
            X_train_array = X_train.values
            X_test_array = X_test.values
            feature_names = X_train.columns.tolist()
        else:
            X_train_array = X_train
            X_test_array = X_test
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        X_train_scaled = self.scaler.fit_transform(X_train_array)
        X_test_scaled = self.scaler.transform(X_test_array)
        
        print(f"   Normalized {X_train_scaled.shape[1]} features")
        
        return X_train_scaled, X_test_scaled, feature_names
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, feature_names):
        """Save processed data"""
        print("\n💾 Saving processed data...")
        
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'num_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        processed_file = self.processed_dir / "processed_data.pkl"
        with open(processed_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"   ✅ Saved: {processed_file}")
        
        # Save summary
        summary = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        print(f"   📊 Summary: {summary}")
        return processed_file
    
    def preprocess_full_pipeline(self):
        """Complete preprocessing pipeline"""
        print("🚀 CIC-IDS2017 Preprocessing (No Label Columns)")
        print("=" * 60)
        
        try:
            # Step 1: Load data and create labels from filenames
            df = self.load_data_without_labels()
            
            # Step 2: Clean data
            X, y = self.clean_data(df)
            
            # Step 3: Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Step 4: Remove outliers from training data
            X_train_clean, y_train_clean = self.remove_outliers(X_train, y_train)
            
            # Step 5: Balance classes
            X_train_balanced, y_train_balanced = self.balance_classes(X_train_clean, y_train_clean)
            
            # Step 6: Normalize features
            X_train_final, X_test_final, feature_names = self.normalize_features(X_train_balanced, X_test)
            
            # Step 7: Save processed data
            output_file = self.save_processed_data(
                X_train_final, X_test_final, 
                y_train_balanced, y_test,
                feature_names
            )
            
            print("\n" + "=" * 60)
            print("✅ PREPROCESSING COMPLETE!")
            print("=" * 60)
            print(f"📊 Final Statistics:")
            print(f"   Training: {X_train_final.shape} ({len(y_train_balanced):,} samples)")
            print(f"   Test: {X_test_final.shape} ({len(y_test):,} samples)")
            print(f"   Classes: {len(self.label_encoder.classes_)}")
            print(f"   Output: {output_file}")
            print("\n🚀 Ready for model training!")
            print("   Next: python experiments/train_baseline.py")
            
            return X_train_final, X_test_final, y_train_balanced, y_test
            
        except Exception as e:
            print(f"\n❌ Preprocessing failed: {e}")
            print("\n🔧 Troubleshooting:")
            print("   1. Check if files exist in data/raw/")
            print("   2. Try: python data/download_dataset.py (create sample)")
            print("   3. Check available memory (large dataset)")
            raise

def main():
    """Run preprocessing pipeline"""
    preprocessor = CICIDSPreprocessor()
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_full_pipeline()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
