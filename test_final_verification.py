"""
Final verification test to confirm the apply_smart_encoding bug fix is working
"""
import pandas as pd
from edaflow.analysis.core import apply_smart_encoding

# Create test data
data = {
    'category_col': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A'],
    'numeric_col': [1, 2, 3, 4, 5, 6, 7, 8],
    'other_col': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y']
}
df = pd.DataFrame(data)

print("🧪 Final Verification Test")
print("=" * 50)
print("Original DataFrame:")
print(df)
print("\n" + "=" * 50)

# Test the specific case that was causing KeyError
print("🔍 Testing apply_smart_encoding with nonexistent target column...")
try:
    result = apply_smart_encoding(
        df=df, 
        target_column='nonexistent_target'  # This should trigger the warning but not error
    )
    print("✅ SUCCESS: Function completed without KeyError!")
    print("📊 Result shape:", result.shape)
    print("📊 Columns after encoding:", list(result.columns))
    print("\n🎉 BUG FIX VERIFIED: The apply_smart_encoding function now handles missing target columns gracefully!")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")
    print("🚨 Bug fix failed!")
