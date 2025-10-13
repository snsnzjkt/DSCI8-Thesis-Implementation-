#!/usr/bin/env python3
"""
Debug script for Wednesday dataset loading issue
"""
import pandas as pd
import numpy as np
from pathlib import Path

def debug_wednesday_file():
    """Debug the Wednesday dataset file specifically"""
    print("🔍 Debugging Wednesday dataset file...")
    
    # Locate the file
    data_dir = Path("data/raw")
    wednesday_file = data_dir / "Wednesday-workingHours.pcap_ISCX.csv"
    
    if not wednesday_file.exists():
        print(f"❌ File not found: {wednesday_file}")
        return
    
    print(f"📁 File found: {wednesday_file}")
    print(f"📏 File size: {wednesday_file.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Try reading the file
        print("\n1️⃣ Reading file...")
        df = pd.read_csv(wednesday_file)
        print(f"   ✅ Loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
        
        # Check basic info
        print("\n2️⃣ Checking data quality...")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
        print(f"   Missing values: {df.isnull().sum().sum():,}")
        print(f"   Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum():,}")
        
        # Check for problematic columns
        print("\n3️⃣ Checking numeric columns...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"   Numeric columns: {len(numeric_cols)}")
        
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"   - {col}: min={vals.min():.3f}, max={vals.max():.3f}, unique={vals.nunique()}")
                
                # Check for extremely small or large values that could cause probability issues
                if vals.min() < -1e10 or vals.max() > 1e10:
                    print(f"     ⚠️  Extreme values detected in {col}")
                    
                if vals.nunique() == 1:
                    print(f"     ⚠️  Constant column detected: {col}")
        
        # Check if there are any string columns that might interfere
        print("\n4️⃣ Checking string columns...")
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            unique_vals = df[col].nunique()
            print(f"   - {col}: {unique_vals} unique values")
            if unique_vals < 20:  # Show values for small cardinality
                print(f"     Values: {list(df[col].unique()[:10])}")
        
        # Try to identify the issue with probabilities
        print("\n5️⃣ Looking for probability-related issues...")
        
        # Check if any column sums to something close to 1 (might be probabilities)
        for col in numeric_cols:
            col_sum = df[col].sum()
            if 0.9 <= abs(col_sum) <= 1.1:
                print(f"   ⚠️  Column {col} sums to {col_sum:.6f} (might be probabilities)")
                
        # Check for rows that might contain probabilities
        for i, row in df.head(10).iterrows():
            row_sum = row.select_dtypes(include=[np.number]).sum()
            if 0.9 <= row_sum <= 1.1:
                print(f"   ⚠️  Row {i} sums to {row_sum:.6f} (might be probabilities)")
                break
    
        # Save a sample for inspection
        sample_file = Path("data/wednesday_sample.csv")
        df.head(1000).to_csv(sample_file, index=False)
        print(f"\n💾 Saved sample to {sample_file}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try reading just the header
        try:
            print("\n🔍 Trying to read just the header...")
            header = pd.read_csv(wednesday_file, nrows=0)
            print(f"   Columns ({len(header.columns)}): {list(header.columns)}")
        except Exception as header_e:
            print(f"   ❌ Can't even read header: {header_e}")
        
        return None

def fix_wednesday_data():
    """Attempt to fix common issues with the Wednesday dataset"""
    print("\n🔧 Attempting to fix Wednesday dataset...")
    
    data_dir = Path("data/raw")
    wednesday_file = data_dir / "Wednesday-workingHours.pcap_ISCX.csv"
    
    try:
        # Read with error handling
        print("📖 Reading file with error handling...")
        df = pd.read_csv(wednesday_file, 
                        error_bad_lines=False,  # Skip bad lines
                        warn_bad_lines=True,    # Warn about bad lines
                        low_memory=False)       # Don't use low memory mode
        
        print(f"   ✅ Read {len(df):,} rows")
        
        # Clean the data
        print("🧹 Cleaning data...")
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        print(f"   Removed empty rows: {len(df):,} remaining")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Remove constant columns (they don't provide information)
        constant_cols = []
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"   Removing {len(constant_cols)} constant columns: {constant_cols[:5]}...")
            df = df.drop(columns=constant_cols)
        
        # Save cleaned version
        cleaned_file = data_dir / "Wednesday-workingHours_cleaned.pcap_ISCX.csv"
        df.to_csv(cleaned_file, index=False)
        print(f"💾 Saved cleaned version to: {cleaned_file}")
        
        return df, cleaned_file
        
    except Exception as e:
        print(f"❌ Failed to fix data: {e}")
        return None, None

if __name__ == "__main__":
    print("🚀 Starting Wednesday dataset debugging...")
    
    # Debug the original file
    df = debug_wednesday_file()
    
    if df is None:
        print("\n⚠️  Original file has issues, attempting to fix...")
        cleaned_df, cleaned_file = fix_wednesday_data()
        
        if cleaned_df is not None:
            print(f"\n✅ Successfully created cleaned version!")
            print(f"📊 Cleaned dataset: {len(cleaned_df):,} rows, {len(cleaned_df.columns)} columns")
        else:
            print("\n❌ Unable to fix the dataset automatically")
            print("\n💡 Suggestions:")
            print("   1. Re-download the Wednesday dataset from the original source")
            print("   2. Check for file corruption")
            print("   3. Try opening the file in a text editor to check for formatting issues")
    else:
        print("\n✅ File loaded successfully, the issue might be in the preprocessing pipeline")