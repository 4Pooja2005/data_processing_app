"""
Type Detection Module
Implements regex-based column type detection using heuristics
"""

import pandas as pd
import re
from typing import Dict, List, Any
from collections import Counter


class SemanticDetector:
    """
    Detects column types using regex patterns and heuristics
    """
    
    def __init__(self):
        # Define regex patterns for different data types
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[1-9][\d]{0,15}$|^\(?[\d]{3}\)?[-\s\.]?[\d]{3}[-\s\.]?[\d]{4}$',
            'date': r'^(0[1-9]|1[0-2])[-/](0[1-9]|[12][0-9]|3[01])[-/]\d{4}$|^\d{4}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12][0-9]|3[01])$',
            'currency': r'^\$?\s*[\d,]+\.?\d*$|^\$?\s*[\d]+\.?\d*\s*USD?$|^\$?\s*[\d,]+\.?\d*\s*USD?$',
            'pincode': r'^\d{5}(-\d{4})?$|^\d{6}$|^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$',
            'numeric': r'^-?\d+\.?\d*$'
        }
    
    def _is_valid_pattern(self, value: str, pattern: str) -> bool:
        """
        Check if value matches the given pattern
        
        Args:
            value: String value to check
            pattern: Regex pattern to match
            
        Returns:
            True if value matches pattern
        """
        if pd.isna(value) or value == '':
            return False
        
        try:
            return bool(re.match(pattern, str(value).strip()))
        except:
            return False
    
    def _detect_email(self, sample_values: List[str]) -> float:
        """Detect if column contains email addresses"""
        matches = sum(1 for val in sample_values if self._is_valid_pattern(val, self.patterns['email']))
        return matches / len(sample_values) if sample_values else 0
    
    def _detect_phone(self, sample_values: List[str]) -> float:
        """Detect if column contains phone numbers"""
        matches = sum(1 for val in sample_values if self._is_valid_pattern(val, self.patterns['phone']))
        return matches / len(sample_values) if sample_values else 0
    
    def _detect_date(self, sample_values: List[str]) -> float:
        """Detect if column contains dates"""
        matches = sum(1 for val in sample_values if self._is_valid_pattern(val, self.patterns['date']))
        return matches / len(sample_values) if sample_values else 0
    
    def _detect_currency(self, sample_values: List[str]) -> float:
        """Detect if column contains currency values"""
        matches = sum(1 for val in sample_values if self._is_valid_pattern(val, self.patterns['currency']))
        return matches / len(sample_values) if sample_values else 0
    
    def _detect_pincode(self, sample_values: List[str]) -> float:
        """Detect if column contains pincodes/ZIP codes"""
        matches = sum(1 for val in sample_values if self._is_valid_pattern(val, self.patterns['pincode']))
        return matches / len(sample_values) if sample_values else 0
    
    def _detect_numeric(self, sample_values: List[str]) -> float:
        """Detect if column contains numeric values"""
        matches = sum(1 for val in sample_values if self._is_valid_pattern(val, self.patterns['numeric']))
        return matches / len(sample_values) if sample_values else 0
    
    def _get_sample_values(self, column_data: pd.Series, sample_size: int = 100) -> List[str]:
        """
        Get sample values from column, excluding NaN and empty values
        
        Args:
            column_data: Pandas Series containing column data
            sample_size: Number of values to sample
            
        Returns:
            List of non-null sample values
        """
        # Remove NaN and empty values
        clean_data = column_data.dropna()
        clean_data = clean_data[clean_data.astype(str).str.strip() != '']
        
        # Sample values (take first N or all if less than N)
        sample_values = clean_data.head(sample_size).astype(str).tolist()
        
        return sample_values
    
    def detect_column_type(self, column_data: pd.Series, column_name: str = "") -> str:
        """
        Detect the most probable type for a column
        
        Args:
            column_data: Pandas Series containing column data
            column_name: Name of the column (for logging/debugging)
            
        Returns:
            Detected type as string
        """
        # Get sample values
        sample_values = self._get_sample_values(column_data)
        
        if not sample_values:
            return 'text'  # Default to text for empty columns
        
        # Calculate confidence scores for each type
        type_scores = {
            'email': self._detect_email(sample_values),
            'phone': self._detect_phone(sample_values),
            'date': self._detect_date(sample_values),
            'currency': self._detect_currency(sample_values),
            'pincode': self._detect_pincode(sample_values),
            'numeric': self._detect_numeric(sample_values)
        }
        
        # Find the type with highest confidence
        max_score = max(type_scores.values())
        
        # Set threshold for type detection (70% match required)
        threshold = 0.7
        
        if max_score >= threshold:
            # Return the type with highest score above threshold
            detected_type = max(type_scores, key=type_scores.get)
            return detected_type
        else:
            # Default to text if no type meets threshold
            return 'text'
    
    def detect_all_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect types for all columns in the DataFrame
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to detected types
        """
        column_types = {}
        
        for column_name in df.columns:
            column_data = df[column_name]
            detected_type = self.detect_column_type(column_data, column_name)
            column_types[column_name] = detected_type
        
        return column_types
    
    def get_type_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of detected types and their distribution
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary with type summary information
        """
        column_types = self.detect_all_column_types(df)
        
        # Count types
        type_counts = Counter(column_types.values())
        
        summary = {
            'total_columns': len(df.columns),
            'column_types': column_types,
            'type_distribution': dict(type_counts),
            'type_percentages': {
                type_name: (count / len(df.columns)) * 100 
                for type_name, count in type_counts.items()
            }
        }
        
        return summary
