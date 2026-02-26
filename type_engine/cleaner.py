"""
Data Preprocessing Module
Implements data cleaning and standardization operations
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional, Union
from datetime import datetime


class AutoCleaner:
    """
    Handles data cleaning and standardization operations
    """
    
    def __init__(self):
        self.operations_log = []
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame
        
        Args:
            df: Input DataFrame
            subset: List of columns to consider for duplicates (None = all columns)
            keep: Which duplicate to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        original_rows = len(df)
        
        try:
            df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
            removed_rows = original_rows - len(df_cleaned)
            
            self.operations_log.append(f"Removed {removed_rows} duplicate rows")
            
            return df_cleaned
            
        except Exception as e:
            raise ValueError(f"Error removing duplicates: {str(e)}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop', columns: Optional[list] = None, fill_value: Any = None) -> pd.DataFrame:
        """
        Handle missing values in DataFrame
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'mean', 'median', 'mode', or 'fill'
            columns: List of columns to process (None = all columns)
            fill_value: Value to use when strategy='fill'
            
        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        
        # Determine which columns to process
        if columns is None:
            columns = df_cleaned.columns.tolist()
        
        try:
            if strategy == 'drop':
                # Drop rows with missing values in specified columns
                original_rows = len(df_cleaned)
                df_cleaned = df_cleaned.dropna(subset=columns)
                removed_rows = original_rows - len(df_cleaned)
                self.operations_log.append(f"Dropped {removed_rows} rows with missing values")
                
            elif strategy == 'mean':
                # Fill with mean for numeric columns only
                for col in columns:
                    if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        mean_val = df_cleaned[col].mean()
                        df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                        self.operations_log.append(f"Filled missing values in '{col}' with mean: {mean_val:.2f}")
                
            elif strategy == 'median':
                # Fill with median for numeric columns only
                for col in columns:
                    if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        median_val = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_val)
                        self.operations_log.append(f"Filled missing values in '{col}' with median: {median_val}")
                
            elif strategy == 'mode':
                # Fill with mode for all columns
                for col in columns:
                    if col in df_cleaned.columns:
                        mode_val = df_cleaned[col].mode()
                        if len(mode_val) > 0:
                            df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                            self.operations_log.append(f"Filled missing values in '{col}' with mode: {mode_val[0]}")
                
            elif strategy == 'fill':
                # Fill with specified value
                if fill_value is None:
                    raise ValueError("fill_value must be provided when strategy='fill'")
                
                for col in columns:
                    if col in df_cleaned.columns:
                        df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                        self.operations_log.append(f"Filled missing values in '{col}' with: {fill_value}")
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            return df_cleaned
            
        except Exception as e:
            raise ValueError(f"Error handling missing values: {str(e)}")
    
    def handle_invalid_values(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """
        Handle invalid values based on detected column types
        
        Args:
            df: Input DataFrame
            column_types: Dictionary mapping column names to their detected types
            
        Returns:
            DataFrame with invalid values handled
        """
        df_cleaned = df.copy()
        
        try:
            for column_name, column_type in column_types.items():
                if column_name not in df_cleaned.columns:
                    continue
                
                column_data = df_cleaned[column_name]
                
                if column_type == 'numeric':
                    # Convert non-numeric values to NaN
                    df_cleaned[column_name] = pd.to_numeric(column_data, errors='coerce')
                    invalid_count = column_data.isna().sum() - df_cleaned[column_name].isna().sum()
                    if invalid_count > 0:
                        self.operations_log.append(f"Converted {invalid_count} invalid values to NaN in '{column_name}'")
                
                elif column_type == 'date':
                    # Try to convert to datetime, invalid values become NaT
                    df_cleaned[column_name] = pd.to_datetime(column_data, errors='coerce')
                    invalid_count = column_data.isna().sum() - df_cleaned[column_name].isna().sum()
                    if invalid_count > 0:
                        self.operations_log.append(f"Converted {invalid_count} invalid date values to NaT in '{column_name}'")
                
                elif column_type == 'email':
                    # Mark invalid emails as NaN
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    invalid_mask = ~column_data.astype(str).str.match(email_pattern, na=False)
                    invalid_count = invalid_mask.sum()
                    if invalid_count > 0:
                        df_cleaned.loc[invalid_mask, column_name] = np.nan
                        self.operations_log.append(f"Marked {invalid_count} invalid emails as NaN in '{column_name}'")
                
                elif column_type == 'phone':
                    # Clean phone numbers and mark invalid ones as NaN
                    phone_pattern = r'^[\+]?[1-9][\d]{0,15}$|^\(?[\d]{3}\)?[-\s\.]?[\d]{3}[-\s\.]?[\d]{4}$'
                    # Remove common formatting first
                    cleaned_phones = column_data.astype(str).str.replace(r'[\s\-\(\)\.]', '', regex=True)
                    invalid_mask = ~cleaned_phones.str.match(phone_pattern, na=False)
                    invalid_count = invalid_mask.sum()
                    if invalid_count > 0:
                        df_cleaned.loc[invalid_mask, column_name] = np.nan
                        self.operations_log.append(f"Marked {invalid_count} invalid phone numbers as NaN in '{column_name}'")
            
            return df_cleaned
            
        except Exception as e:
            raise ValueError(f"Error handling invalid values: {str(e)}")
    
    def standardize_dates(self, df: pd.DataFrame, date_columns: Optional[list] = None, target_format: str = 'YYYY-MM-DD') -> pd.DataFrame:
        """
        Standardize date formats
        
        Args:
            df: Input DataFrame
            date_columns: List of date columns to standardize (None = auto-detect)
            target_format: Target date format
            
        Returns:
            DataFrame with standardized dates
        """
        df_cleaned = df.copy()
        
        try:
            if date_columns is None:
                # Auto-detect date columns
                date_columns = []
                for col in df_cleaned.columns:
                    # Try to convert to datetime to check if it's a date column
                    sample = df_cleaned[col].dropna().head(10)
                    if len(sample) > 0:
                        try:
                            pd.to_datetime(sample, errors='raise')
                            date_columns.append(col)
                        except:
                            continue
            
            for col in date_columns:
                if col in df_cleaned.columns:
                    # Convert to datetime
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    
                    # Format to target format
                    if target_format == 'YYYY-MM-DD':
                        df_cleaned[col] = df_cleaned[col].dt.strftime('%Y-%m-%d')
                    elif target_format == 'DD-MM-YYYY':
                        df_cleaned[col] = df_cleaned[col].dt.strftime('%d-%m-%Y')
                    elif target_format == 'MM-DD-YYYY':
                        df_cleaned[col] = df_cleaned[col].dt.strftime('%m-%d-%Y')
                    
                    self.operations_log.append(f"Standardized dates in '{col}' to {target_format}")
            
            return df_cleaned
            
        except Exception as e:
            raise ValueError(f"Error standardizing dates: {str(e)}")
    
    def standardize_numbers(self, df: pd.DataFrame, numeric_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Standardize number formatting (remove commas, normalize decimals)
        
        Args:
            df: Input DataFrame
            numeric_columns: List of numeric columns to standardize (None = auto-detect)
            
        Returns:
            DataFrame with standardized numbers
        """
        df_cleaned = df.copy()
        
        try:
            if numeric_columns is None:
                # Auto-detect numeric columns
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_columns:
                if col in df_cleaned.columns:
                    # Convert to string, remove commas, then convert back to numeric
                    df_cleaned[col] = df_cleaned[col].astype(str).str.replace(',', '', regex=False)
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    
                    self.operations_log.append(f"Standardized number formatting in '{col}'")
            
            return df_cleaned
            
        except Exception as e:
            raise ValueError(f"Error standardizing numbers: {str(e)}")
    
    def standardize_text(self, df: pd.DataFrame, text_columns: Optional[list] = None, case: str = 'lower') -> pd.DataFrame:
        """
        Standardize text formatting
        
        Args:
            df: Input DataFrame
            text_columns: List of text columns to standardize (None = auto-detect non-numeric)
            case: 'lower', 'upper', or 'title'
            
        Returns:
            DataFrame with standardized text
        """
        df_cleaned = df.copy()
        
        try:
            if text_columns is None:
                # Auto-detect text columns (non-numeric)
                text_columns = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
            
            for col in text_columns:
                if col in df_cleaned.columns:
                    # Convert to string first
                    df_cleaned[col] = df_cleaned[col].astype(str)
                    
                    # Apply case transformation
                    if case == 'lower':
                        df_cleaned[col] = df_cleaned[col].str.lower()
                    elif case == 'upper':
                        df_cleaned[col] = df_cleaned[col].str.upper()
                    elif case == 'title':
                        df_cleaned[col] = df_cleaned[col].str.title()
                    
                    # Trim whitespace
                    df_cleaned[col] = df_cleaned[col].str.strip()
                    
                    self.operations_log.append(f"Standardized text in '{col}' to {case} case")
            
            return df_cleaned
            
        except Exception as e:
            raise ValueError(f"Error standardizing text: {str(e)}")
    
    def get_operations_log(self) -> list:
        """
        Get the log of all operations performed
        
        Returns:
            List of operation descriptions
        """
        return self.operations_log.copy()
    
    def clear_operations_log(self):
        """Clear the operations log"""
        self.operations_log.clear()
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with DataFrame summary
        """
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        return summary
