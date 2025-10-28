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
    print("="*70)
    print("REPROCESSING CIC-IDS2017 DATASET")
    print("="*70)
    print(f"Target Features: {config.NUM_FEATURES}")
    print(f"Clearing existing preprocessed files...")
    
    preprocessor = CICIDSPreprocessor()
    
    # Force reprocess and validate feature count
    results = preprocessor.process_all_files(force_reprocess=True, validate_features=True)
    
    print("\nValidation Results:")
    print(f"Features processed: {results['num_features']}")
    if results['num_features'] != config.NUM_FEATURES:
        print(f"⚠️ WARNING: Expected {config.NUM_FEATURES} features but got {results['num_features']}")
        sys.exit(1)
    print("✓ Feature count matches configuration")
    
if __name__ == "__main__":
    main()