import sys
import os

# Add the project root to the python path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlp.engine import NLPEngine

def run_tests():
    engine = NLPEngine()
    
    test_cases = [
        # Remove Duplicates
        ("Please remove the duplicates from my data.", "remove_duplicates"),
        ("drop duplicates", "remove_duplicates"),
        ("I need to delete the duplicates now.", "remove_duplicates"),
        ("Clean duplicates please", "remove_duplicates"),
        ("Can you deduplicate this file?", "remove_duplicates"),
        
        # Drop Missing
        ("remove missing values", "drop_missing"),
        ("drop empty rows", "drop_missing"),
        ("delete the null data", "drop_missing"),
        ("get rid of missing rows", "drop_missing"),
        ("Clean the empty rows out", "drop_missing"),
        ("drop na", "drop_missing"),
        
        # Fill Zeros
        ("fill the missing values with zero", "fill_zero_missing"),
        ("replace empty rows with zeros", "fill_zero_missing"),
        ("zero out the null data", "fill_zero_missing"),
        ("Set missing to zero", "fill_zero_missing"),
        
        # Standardize Strings
        ("Convert these names to lowercase", "standardize_lower"),
        ("Please lowercase the emails", "standardize_lower"),
        ("Make everything uppercase", "standardize_upper"),
        ("uppercase all symbols", "standardize_upper"),
        ("Convert titles to title case", "standardize_title"),
        ("strip the extra whitespace", "standardize_strip"),
        ("trim spaces", "standardize_strip"),
        
        # Dates and Currency
        ("format the dates please", "standardize_date"),
        ("standardize time format", "standardize_date"),
        ("remove the currency symbols", "remove_currency"),
        ("clean the prices", "remove_currency"),
        
        # Fails / Unrecognized
        ("Summarize the data for me", None),
        ("Make a chart of the sales", None),
        ("Who is the best employee?", None)
    ]
    
    passed = 0
    total = len(test_cases)
    
    print("=== NLP Engine Test Suite ===")
    print(f"Testing {total} prompts...\n")
    
    for prompt, expected_intent in test_cases:
        result = engine.parse_command(prompt)
        actual_intent = result["intent"] if result else None
        
        if actual_intent == expected_intent:
            passed += 1
            status = "PASS"
        else:
            status = f"FAIL (Expected: {expected_intent}, Got: {actual_intent})"
            
        print(f"[{status}] \"{prompt}\" -> {actual_intent}")
        
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    run_tests()
