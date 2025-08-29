# data/download_dataset.py - CIC-IDS2017 Dataset Downloader
import os
import requests
import zipfile
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd

class CICIDSDownloader:
    """Download and setup CIC-IDS2017 dataset"""
    
    def __init__(self):
        self.base_dir = Path("data")
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset files information
        self.dataset_files = {
            "Monday-WorkingHours.pcap_ISCX.csv": {
                "description": "Benign traffic only",
                "size_mb": 532
            },
            "Tuesday-WorkingHours.pcap_ISCX.csv": {
                "description": "Benign + FTP-Patator + SSH-Patator",
                "size_mb": 428
            },
            "Wednesday-workingHours.pcap_ISCX.csv": {
                "description": "Benign + DoS attacks + Heartbleed", 
                "size_mb": 440
            },
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": {
                "description": "Benign + Web attacks",
                "size_mb": 169
            },
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": {
                "description": "Benign + Infiltration",
                "size_mb": 113
            },
            "Friday-WorkingHours-Morning.pcap_ISCX.csv": {
                "description": "Benign + Botnet",
                "size_mb": 189
            },
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": {
                "description": "Benign + Port Scan",
                "size_mb": 204
            },
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": {
                "description": "Benign + DDoS attacks",
                "size_mb": 379
            }
        }
    
    def check_existing_files(self):
        """Check which files already exist"""
        existing_files = []
        missing_files = []
        
        for filename in self.dataset_files.keys():
            filepath = self.raw_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                existing_files.append((filename, size_mb))
            else:
                missing_files.append(filename)
        
        return existing_files, missing_files
    
    def print_dataset_status(self):
        """Print current dataset status"""
        existing_files, missing_files = self.check_existing_files()
        
        print("="*60)
        print("CIC-IDS2017 Dataset Status")
        print("="*60)
        
        if existing_files:
            print(f"\n✅ FOUND ({len(existing_files)} files):")
            for filename, size_mb in existing_files:
                desc = self.dataset_files[filename]["description"]
                print(f"   📁 {filename}")
                print(f"      Size: {size_mb:.1f} MB - {desc}")
        
        if missing_files:
            print(f"\n❌ MISSING ({len(missing_files)} files):")
            for filename in missing_files:
                desc = self.dataset_files[filename]["description"]
                expected_size = self.dataset_files[filename]["size_mb"]
                print(f"   📁 {filename}")
                print(f"      Expected: {expected_size} MB - {desc}")
        
        total_size_gb = sum(info["size_mb"] for info in self.dataset_files.values()) / 1024
        print(f"\n📊 Total dataset size: ~{total_size_gb:.1f} GB")
        
        if not missing_files:
            print("\n🎉 All files are present! Ready for preprocessing.")
            return True
        else:
            print(f"\n⚠️  Please download {len(missing_files)} missing files.")
            return False
    
    def print_download_instructions(self):
        """Print manual download instructions"""
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print("\n🌐 Official Source:")
        print("   https://www.unb.ca/cic/datasets/ids-2017.html")
        print("\n📋 Steps:")
        print("   1. Visit the URL above")
        print("   2. Scroll to 'Download Dataset' section")
        print("   3. Download all CSV files")
        print(f"   4. Place files in: {self.raw_dir.absolute()}")
        
        print("\n🔄 Alternative: Kaggle")
        print("   https://www.kaggle.com/datasets/cicdataset/cicids2017")
        
        print("\n📁 Required files:")
        for filename, info in self.dataset_files.items():
            print(f"   - {filename} ({info['size_mb']} MB)")
    
    def create_sample_dataset(self):
        """Create a small sample for testing purposes"""
        print("\n🧪 Creating sample dataset for testing...")
        
        sample_file = self.raw_dir / "sample_cicids2017.csv"
        
        if sample_file.exists():
            print(f"   ✅ Sample file already exists: {sample_file}")
            return True
        
        try:
            # Create synthetic data that mimics CIC-IDS2017 structure
            import numpy as np
            import pandas as pd
            
            # Set random seed for reproducibility
            np.random.seed(42)
            
            # CIC-IDS2017 feature names (partial list for sample)
            feature_names = [
                'Destination Port', 'Flow Duration', 'Total Fwd Packets',
                'Total Backward Packets', 'Total Length of Fwd Packets',
                'Total Length of Bwd Packets', 'Fwd Packet Length Max',
                'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                'Fwd Packet Length Std', 'Bwd Packet Length Max',
                'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
                'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min'
            ]
            
            # Extend to 78 features total
            while len(feature_names) < 78:
                feature_names.append(f'Feature_{len(feature_names)}')
            
            # Generate sample data
            n_samples = 5000
            data = {}
            
            # Generate realistic network traffic features
            for feature in feature_names:
                if 'Port' in feature:
                    data[feature] = np.random.choice([80, 443, 22, 21, 53], n_samples)
                elif 'Duration' in feature:
                    data[feature] = np.random.exponential(1000, n_samples)
                elif 'Packet' in feature:
                    data[feature] = np.random.poisson(10, n_samples)
                elif 'Length' in feature:
                    data[feature] = np.random.gamma(2, 500, n_samples)
                else:
                    data[feature] = np.random.normal(0, 1, n_samples)
            
            # Add labels with realistic distribution
            attack_types = [
                'BENIGN', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
                'DoS slowloris', 'FTP-Patator', 'SSH-Patator', 'Web Attack – Brute Force',
                'Web Attack – Sql Injection', 'Web Attack – XSS', 'Infiltration',
                'Bot', 'PortScan', 'Heartbleed'
            ]
            
            # 80% benign, 20% attacks (realistic distribution)
            label_probs = [0.8] + [0.2/14] * 14
            data['Label'] = np.random.choice(attack_types, n_samples, p=label_probs)
            
            # Create DataFrame and save
            df = pd.DataFrame(data)
            df.to_csv(sample_file, index=False)
            
            print(f"   ✅ Created sample dataset: {sample_file}")
            print(f"   📊 Sample size: {len(df)} records, {len(df.columns)} features")
            print(f"   🏷️  Labels: {df['Label'].value_counts().head()}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating sample: {e}")
            return False
    
    def validate_dataset_files(self):
        """Validate downloaded dataset files"""
        existing_files, missing_files = self.check_existing_files()
        
        if not existing_files:
            print("❌ No dataset files found.")
            return False
        
        print("\n🔍 Validating dataset files...")
        valid_files = []
        
        for filename, size_mb in existing_files:
            filepath = self.raw_dir / filename
            try:
                # Try to read first few rows to validate format
                df_sample = pd.read_csv(filepath, nrows=5)
                
                if 'Label' in df_sample.columns:
                    print(f"   ✅ {filename} - Valid CSV with Label column")
                    valid_files.append(filename)
                else:
                    print(f"   ⚠️  {filename} - Missing 'Label' column")
                
            except Exception as e:
                print(f"   ❌ {filename} - Error reading file: {e}")
        
        if valid_files:
            print(f"\n🎉 {len(valid_files)} valid dataset files ready for use!")
            return True
        else:
            print("\n❌ No valid dataset files found.")
            return False

def main():
    """Main function to handle dataset download and setup"""
    print("🚀 CIC-IDS2017 Dataset Setup")
    
    downloader = CICIDSDownloader()
    
    # Check current status
    files_ready = downloader.print_dataset_status()
    
    if files_ready:
        # Validate files
        if downloader.validate_dataset_files():
            print("\n✅ Dataset is ready for preprocessing!")
            print("\n🔄 Next step: Run 'python data/preprocess.py'")
        else:
            print("\n⚠️  Files exist but may be corrupted. Please re-download.")
    else:
        # Print download instructions
        downloader.print_download_instructions()
        
        # Offer to create sample for testing
        print("\n" + "="*60)
        print("TESTING OPTION")
        print("="*60)
        print("Would you like to create a sample dataset for testing?")
        print("(This won't replace the real dataset)")
        
        response = input("\nCreate sample dataset? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if downloader.create_sample_dataset():
                print("\n✅ Sample dataset created!")
                print("🔄 You can now test with: python data/preprocess.py")
                print("⚠️  Remember to replace with real CIC-IDS2017 data later!")

if __name__ == "__main__":
    main()
