#!/usr/bin/env python3
"""
Fix for the Wednesday dataset preprocessing issue
This script will patch the preprocess.py file to handle infinite values properly
"""

def fix_preprocessing_pipeline():
    """Add better infinite value handling to the preprocessing pipeline"""
    
    # Read the current preprocessing file
    preprocess_file = "data/preprocess.py"
    
    with open(preprocess_file, 'r') as f:
        content = f.read()
    
    # Check if the fix is already applied
    if "handle_infinite_values" in content:
        print("✅ Infinite value handling already exists in preprocessing pipeline")
        return
    
    # Find the place to add the infinite value handler
    # We'll add it right after the basic data loading
    
    fix_code = '''
    def handle_infinite_values(self, df):
        """Handle infinite and problematic values in the dataset"""
        print("🔧 Handling infinite and extreme values...")
        
        initial_rows = len(df)
        
        # Replace infinite values with NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
                
        if inf_counts:
            print(f"   Found infinite values in {len(inf_counts)} columns:")
            for col, count in list(inf_counts.items())[:5]:  # Show first 5
                print(f"     - {col}: {count} infinite values")
            if len(inf_counts) > 5:
                print(f"     ... and {len(inf_counts) - 5} more columns")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle extreme values (very large numbers that might cause numerical issues)
        for col in numeric_cols:
            if col in df.columns:  # Check if column still exists
                values = df[col].dropna()
                if len(values) > 0:
                    # Calculate reasonable bounds (99.9th percentile)
                    upper_bound = values.quantile(0.999)
                    lower_bound = values.quantile(0.001)
                    
                    # Cap extreme values
                    extreme_mask = (df[col] > upper_bound) | (df[col] < lower_bound)
                    extreme_count = extreme_mask.sum()
                    
                    if extreme_count > 0:
                        df.loc[df[col] > upper_bound, col] = upper_bound
                        df.loc[df[col] < lower_bound, col] = lower_bound
                        
                        if extreme_count > 100:  # Only report for significant numbers
                            print(f"   Capped {extreme_count} extreme values in {col}")
        
        # Fill remaining NaN values with median (more robust than mean)
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                if not pd.isna(median_val):
                    df[col].fillna(median_val, inplace=True)
                else:
                    # If median is NaN, use 0
                    df[col].fillna(0, inplace=True)
        
        # Remove constant columns (they don't provide information and can cause issues)
        constant_cols = []
        for col in numeric_cols:
            if col in df.columns and df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"   Removing {len(constant_cols)} constant columns")
            df = df.drop(columns=constant_cols)
        
        final_rows = len(df)
        print(f"   Processed data: {final_rows:,} rows ({initial_rows - final_rows:,} removed)")
        
        return df
'''
    
    # Insert the method after the class definition but before other methods
    class_def_pos = content.find("def __init__(self):")
    if class_def_pos == -1:
        print("❌ Could not find class definition in preprocess.py")
        return
    
    # Find the end of __init__ method
    init_end = content.find("\n    def ", class_def_pos + 1)
    if init_end == -1:
        init_end = len(content)
    
    # Insert our new method
    new_content = content[:init_end] + fix_code + content[init_end:]
    
    # Now we need to modify the load_data method to call our infinite handler
    # Find the load_data method and add our call
    
    load_data_pattern = "df_combined = pd.concat(all_dataframes, ignore_index=True)"
    if load_data_pattern in new_content:
        replacement = f"""{load_data_pattern}
        
        # Handle infinite values before further processing
        df_combined = self.handle_infinite_values(df_combined)"""
        
        new_content = new_content.replace(load_data_pattern, replacement)
    
    # Write the updated file
    with open(preprocess_file, 'w') as f:
        f.write(new_content)
    
    print("✅ Added infinite value handling to preprocessing pipeline")
    print("💡 The preprocessing should now handle the Wednesday dataset properly")

if __name__ == "__main__":
    print("🔧 Fixing preprocessing pipeline for Wednesday dataset...")
    fix_preprocessing_pipeline()
    print("\n🚀 Try running the preprocessing again:")
    print("   python data/preprocess.py")
    print("   or")
    print("   python main.py")