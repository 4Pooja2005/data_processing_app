import re
import pandas as pd
from rapidfuzz import fuzz, process

class SchemaMapper:
    def __init__(self, threshold=70.0):
        """
        Initialize the SchemaMapper.
        :param threshold: Minimum matching score (0-100) to consider a column a match.
        """
        self.threshold = threshold
        self.abbreviations = {
            "cust": "customer",
            "dob": "date of birth",
            "num": "number",
            "#": "number",
            "addr": "address",
            "tel": "telephone",
            "zip": "zipcode",
            "postcode": "postal code",
            "usr": "user",
            "nm": "name"
        }

    def _preprocess(self, text):
        text = str(text)
        # Split CamelCase/PascalCase
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Replace underscores and hyphens with spaces
        text = text.replace("_", " ").replace("-", " ")
        # Expand abbreviations
        words = text.split()
        expanded = [self.abbreviations.get(w, w) for w in words]
        text = " ".join(expanded)
        # Handle special characters (like phone# -> phone number)
        for abbr, full in self.abbreviations.items():
            if not abbr.isalpha():
                text = text.replace(abbr, " " + full + " ")
        # Clean extra spaces
        return re.sub(r'\s+', ' ', text).strip()

    def match_columns(self, source_cols, target_cols):
        """
        Takes a list of source columns and target (golden standard) columns.
        Returns a dictionary mapping {source_col: target_col} for matches above the threshold.
        """
        mapping = {}
        target_pool = list(target_cols)

        # Precompute preprocessed targets
        prep_targets = {self._preprocess(t): t for t in target_pool}

        for s_col in source_cols:
            if not target_pool:
                break
            
            prep_source = self._preprocess(s_col)
            
            # Match against preprocessed targets
            result = process.extractOne(prep_source, list(prep_targets.keys()), scorer=fuzz.token_set_ratio)
            
            if result:
                best_match_prep, score, _ = result
                if score >= self.threshold:
                    original_target = prep_targets[best_match_prep]
                    # Only map if target is still available
                    if original_target in target_pool:
                        mapping[s_col] = original_target
                        target_pool.remove(original_target)
                        del prep_targets[best_match_prep]
        
        return mapping

    def apply_schema(self, df, golden_cols):
        """
        Applies the golden schema to the dataframe.
        Renames columns that fuzzy match the golden standard.
        """
        mapping = self.match_columns(df.columns, golden_cols)
        mapped_df = df.rename(columns=mapping)
        return mapped_df, mapping
