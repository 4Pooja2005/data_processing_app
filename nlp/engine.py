import re

class NLPEngine:
    def __init__(self):
        # We define intent categories and keyword aliases.
        self.intents = {
            "remove_duplicates": [
                r"\b(remove|drop|delete|clear|clean|get rid of) (the )?duplicates\b",
                r"\bdeduplicate\b"
            ],
            "drop_missing": [
                r"\b(remove|drop|delete|clear|clean|get rid of) (the )?(null|empty|missing) (rows|values|data)?\b",
                r"\bdrop (nan|na)\b"
            ],
            "fill_zero_missing": [
                r"\b(fill|replace) (the )?(null|empty|missing) (rows|values|data)? with zeros?\b",
                r"\bzero out (the )?(null|empty|missing) (rows|values|data)?\b",
                r"\bset (null|missing|empty) to zero\b"
            ],
            "standardize_lower": [
                r"\b(convert|make|transform|change) .* to (lower|lowercase)\b",
                r"\blowercase\b"
            ],
            "standardize_upper": [
                r"\b(convert|make|transform|change) .* to (upper|uppercase|caps)\b",
                r"\buppercase\b"
            ],
            "standardize_title": [
                r"\b(convert|make|transform|change) .* to (title|titlecase) case\b",
                r"\btitlecase\b"
            ],
            "standardize_strip": [
                r"\b(strip|trim|remove) (the )?(leading|trailing|extra)? (whitespace|spaces)\b",
                r"\btrim spaces\b"
            ],
            "standardize_date": [
                r"\b(standardize|format|convert) (the )?(dates?|times?)\b"
            ],
            "remove_currency": [
                r"\b(remove|strip|clean) (the )?(currency|money|dollar|euro|pound) (symbols?|signs?)\b",
                r"\bclean (the )?prices?\b"
            ]
        }

    def parse_command(self, text):
        """
        Parses a natural language command and returns the identified intent.
        Returns None if no intent is found.
        """
        # Lowercase and remove basic punctuation for easier matching
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
        # Clean up multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if re.search(pattern, clean_text):
                    # For column-specific standardizations (lower, upper, etc.), we would ideally
                    # extract the column name. For now, we return the intent and the UI can handle the rest.
                    # Or, if we want to extract simple exact matches for column words:
                    # In a real rigorous setup, we'd do Entity Extraction here.
                    return {
                        "intent": intent,
                        "raw_command": text,
                        "match": pattern
                    }
        
        return None
