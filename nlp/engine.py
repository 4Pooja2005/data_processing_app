import re
from .advanced_parser import NLPCommandParser

class NLPEngine:
    def __init__(self):
        self.advanced_parser = NLPCommandParser()
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

    def parse_command(self, text, df=None):
        """
        Parses a natural language command.
        First attempts the advanced parser if a DataFrame is provided.
        Returns None if no intent is found.
        """
        if df is not None:
            advanced_result = self.advanced_parser.parse_command(text, df)
            if advanced_result and advanced_result.get("success"):
                return {
                    "intent": "advanced_operation",
                    "operation": advanced_result["operation"],
                    "details": advanced_result
                }

        # Lowercase and remove basic punctuation for easier matching
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
        # Clean up multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if re.search(pattern, clean_text):
                    return {
                        "intent": intent,
                        "raw_command": text,
                        "match": pattern
                    }
        
        return None

    def execute_advanced(self, df, details):
        """Applies the parsed advanced operation to the dataframe and returns the new dataframe."""
        op = details.get("operation")
        if not op:
            return df

        try:
            if op == "filter_rows":
                cond = details.get("parsed_condition", {})
                col = cond.get("column")
                operator = cond.get("operator")
                val = cond.get("value")
                
                if operator == "eq": return df[df[col] == val]
                if operator == "ne": return df[df[col] != val]
                if operator == "gt": return df[df[col] > val]
                if operator == "lt": return df[df[col] < val]
                if operator == "ge": return df[df[col] >= val]
                if operator == "le": return df[df[col] <= val]
                if operator == "contains": return df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
                if operator == "startswith": return df[df[col].astype(str).str.startswith(str(val), na=False)]
                if operator == "endswith": return df[df[col].astype(str).str.endswith(str(val), na=False)]
            
            elif op == "filter_columns":
                cols = details.get("columns", [])
                return df[cols]
                
            elif op == "rename_column":
                old = details.get("old_name")
                new = details.get("new_name")
                return df.rename(columns={old: new})
                
            elif op == "delete_column":
                cols = details.get("columns", [])
                return df.drop(columns=cols, errors='ignore')
                
            elif op == "convert_type":
                col = details.get("column")
                t_type = details.get("target_type")
                new_df = df.copy()
                new_df[col] = new_df[col].astype(t_type)
                return new_df
                
            elif op == "sort_by":
                cols = details.get("columns", [])
                ascending = details.get("ascending", [])
                return df.sort_values(by=cols, ascending=ascending)
                
            elif op == "split_column":
                src = details.get("source_column")
                targets = details.get("target_columns", [])
                new_df = df.copy()
                parts = new_df[src].astype(str).str.split(expand=True)
                for i, t in enumerate(targets):
                    if i < len(parts.columns):
                        new_df[t] = parts[i]
                return new_df
                
            elif op == "combine_columns":
                srcs = details.get("source_columns", [])
                target = details.get("target_column")
                new_df = df.copy()
                new_df[target] = new_df[srcs].astype(str).agg(' '.join, axis=1)
                return new_df
                
        except Exception as e:
            print(f"Error executing advanced NLP: {e}")
        
        return df
