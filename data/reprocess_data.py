"""
Quick script to reprocess the CIC-IDS2017 dataset
Ensures we have all 78 features as per thesis requirements
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocess import CICIDSPreprocessor
from config import config

def main():
    print("Reprocessing CIC-IDS2017 dataset...")
    print(f"Target: {config.NUM_FEATURES} features")
    
    preprocessor = CICIDSPreprocessor()
    preprocessor.process_all_files(force_reprocess=True)
    
if __name__ == "__main__":
    main()