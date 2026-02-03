import sys
import os
import pandas as pd

# Add the project root to the path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.processor import standardize_column

def test_num_to_words():
    df = pd.DataFrame({'numbers': [21, 105, 0, 99]})
    print("Testing num_to_words...")
    result = standardize_column(df, 'numbers', 'num_to_words')
    print(result)
    
    expected = ['twenty-one', 'one hundred and five', 'zero', 'ninety-nine']
    for i, val in enumerate(result['numbers']):
        assert val == expected[i], f"Expected {expected[i]}, got {val}"
    print("num_to_words passed successfully!")

def test_words_to_num():
    df = pd.DataFrame({'words': ['twenty-one', 'one hundred and five', 'zero', 'ninety-nine', 'invalid']})
    print("\nTesting words_to_num...")
    result = standardize_column(df, 'words', 'words_to_num')
    print(result)
    
    # word2number handles basic numbers. 'invalid' should remain as is or be handled gracefully
    # Based on implementation: if error, it returns original value
    assert result['words'].iloc[0] == 21
    assert result['words'].iloc[1] == 105
    assert result['words'].iloc[2] == 0
    assert result['words'].iloc[3] == 99
    assert result['words'].iloc[4] == 'invalid'
    print("words_to_num passed successfully!")

if __name__ == "__main__":
    try:
        test_num_to_words()
        test_words_to_num()
        print("\nAll tests passed!")
    except ImportError:
        print("Dependencies not installed correctly.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
