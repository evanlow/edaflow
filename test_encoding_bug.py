#!/usr/bin/env python3
"""
Test script to reproduce and fix the apply_smart_encoding bug
"""

import pandas as pd
import edaflow

def test_encoding_bug():
    """Test case to reproduce the KeyError: 'target' bug"""
    
    # Create a simple test DataFrame without a 'target' column
    df = pd.DataFrame({
        'category_col': ['A', 'B', 'A', 'C', 'B'],
        'numeric_col': [1, 2, 3, 4, 5],
        'binary_col': ['Yes', 'No', 'Yes', 'No', 'Yes']
    })
    
    print("Original DataFrame:")
    print(df)
    print(f"Columns: {list(df.columns)}")
    
    try:
        print("\n🧠 Step 1: Analyzing encoding needs...")
        # This might be setting target_column incorrectly
        analysis = edaflow.analyze_encoding_needs(df, target_column='target')
        print("Analysis completed successfully!")
        print(f"Recommendations: {analysis['recommendations']}")
        print(f"Priority: {analysis['encoding_priority']}")
        
        print("\n⚡ Step 2: Applying smart encoding...")
        # This should fail with KeyError: 'target'
        df_encoded = edaflow.apply_smart_encoding(df, encoding_analysis=analysis, target_column='target')
        print("Encoding completed successfully!")
        print(df_encoded.head())
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_encoding_fix():
    """Test the fix - should work without errors"""
    
    # Create a test DataFrame with a proper target column
    df = pd.DataFrame({
        'category_col': ['A', 'B', 'A', 'C', 'B'],
        'numeric_col': [1, 2, 3, 4, 5],
        'binary_col': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'target': [0, 1, 0, 1, 1]
    })
    
    print("\n" + "="*50)
    print("Testing with proper target column:")
    print(df)
    
    try:
        print("\n🧠 Step 1: Analyzing encoding needs...")
        analysis = edaflow.analyze_encoding_needs(df, target_column='target')
        
        print("\n⚡ Step 2: Applying smart encoding...")
        df_encoded = edaflow.apply_smart_encoding(df, encoding_analysis=analysis, target_column='target')
        print("Encoding completed successfully!")
        print(df_encoded.head())
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing edaflow encoding bug...")
    test_encoding_bug()
    test_encoding_fix()
