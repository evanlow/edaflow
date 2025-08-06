#!/usr/bin/env python3
"""
Test to reproduce the exact error from the screenshot
"""

import pandas as pd
import edaflow

def test_exact_error_scenario():
    """Test the exact scenario from the error screenshot"""
    
    # Create test data similar to what might cause the KeyError
    df = pd.DataFrame({
        'feature1': ['A', 'B', 'A', 'C', 'B'] * 4,  # 4 columns as mentioned in the screenshot
        'feature2': [1, 2, 3, 4, 5] * 4,
        'feature3': ['X', 'Y', 'Z', 'X', 'Y'] * 4,
        'feature4': [0.1, 0.2, 0.3, 0.4, 0.5] * 4
    })
    
    print("Test DataFrame:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
    
    try:
        # Try to reproduce the exact scenario from the error
        print("\n🔍 Running analyze_encoding_needs with 'target' column that doesn't exist...")
        analysis = edaflow.analyze_encoding_needs(df, target_column='target')
        
        print("\n⚡ Running apply_smart_encoding...")
        df_encoded = edaflow.apply_smart_encoding(df, target_column='target')
        
        print("✅ Success - no error occurred!")
        
    except Exception as e:
        print(f"❌ ERROR REPRODUCED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_target_encoding_specific():
    """Test target encoding specifically"""
    
    # Create a DataFrame where target encoding would be recommended
    df = pd.DataFrame({
        'high_card_cat': [f'cat_{i%15}' for i in range(100)],  # High cardinality - should get target encoding
        'feature2': range(100),
    })
    
    print(f"\nTest DataFrame for target encoding:")
    print(f"Shape: {df.shape}")
    print(f"high_card_cat unique values: {df['high_card_cat'].nunique()}")
    
    try:
        # This should recommend target encoding for high_card_cat
        print("\n🔍 Analyzing high cardinality categorical column...")
        analysis = edaflow.analyze_encoding_needs(
            df, 
            target_column='nonexistent_target',  # This target doesn't exist!
            max_cardinality_onehot=5,  # Force high cardinality column to target encoding
            max_cardinality_target=20
        )
        
        print("Analysis results:")
        print(f"Recommendations: {analysis['recommendations']}")
        
        print("\n⚡ Applying smart encoding with nonexistent target...")
        df_encoded = edaflow.apply_smart_encoding(
            df, 
            encoding_analysis=analysis,
            target_column='nonexistent_target'
        )
        
        print("✅ Success!")
        print(f"Final shape: {df_encoded.shape}")
        print(f"Final columns: {list(df_encoded.columns)}")
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing exact error scenarios...")
    test_exact_error_scenario()
    test_target_encoding_specific()
