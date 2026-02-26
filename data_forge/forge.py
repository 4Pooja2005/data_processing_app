"""
Advanced Data Merging Engine
Handles vertical union merges and smart horizontal joins with conflict resolution
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DataForgeEngine:
    """
    Advanced data merging engine with privacy-focused features
    """
    
    def __init__(self):
        self.merge_history = []
        self.conflict_log = []
        
        # Regex patterns for heuristic detection
        self.patterns = {
            'pincode': r'^\d{6}$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[1-9][\d]{0,15}$|^\(?[\d]{3}\)?[-\s\.]?[\d]{3}[-\s\.]?[\d]{4}$',
            'coordinates': r'^-?\d+\.?\d*,\s*-?\d+\.?\d*$'
        }
    
    def vertical_union_merge(self, 
                           df1: pd.DataFrame, 
                           df2: pd.DataFrame,
                           auto_drop_duplicates: bool = True,
                           duplicate_threshold: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform vertical union merge with automatic duplicate detection and removal
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            auto_drop_duplicates: Whether to automatically remove 100% identical rows
            duplicate_threshold: Threshold for duplicate detection (1.0 = 100% identical)
            
        Returns:
            Tuple of (merged DataFrame, merge statistics)
        """
        
        try:
            logger.info(f"Starting vertical union merge: {len(df1)} + {len(df2)} rows")
            
            # Validate input DataFrames
            if df1.empty and df2.empty:
                raise ValueError("Both DataFrames are empty")
            
            # Ensure columns are compatible
            if not df1.empty and not df2.empty:
                common_columns = set(df1.columns) & set(df2.columns)
                if not common_columns:
                    logger.warning("No common columns found between DataFrames")
                
                # Align columns - use union of all columns
                all_columns = list(set(df1.columns) | set(df2.columns))
                
                # Add missing columns with NaN values
                df1_aligned = df1.reindex(columns=all_columns)
                df2_aligned = df2.reindex(columns=all_columns)
            else:
                # If one is empty, use the other's columns
                if df1.empty:
                    all_columns = df2.columns.tolist()
                    df1_aligned = df1.reindex(columns=all_columns)
                    df2_aligned = df2.copy()
                else:
                    all_columns = df1.columns.tolist()
                    df1_aligned = df1.copy()
                    df2_aligned = df2.reindex(columns=all_columns)
            
            # Concatenate DataFrames
            merged_df = pd.concat([df1_aligned, df2_aligned], ignore_index=True)
            
            original_rows = len(merged_df)
            duplicates_removed = 0
            
            # Auto-drop duplicates if requested
            if auto_drop_duplicates and len(merged_df) > 0:
                # Find exact duplicates (100% identical across all columns)
                duplicate_mask = merged_df.duplicated(keep=False)
                duplicate_count = duplicate_mask.sum()
                
                if duplicate_count > 0:
                    logger.info(f"Found {duplicate_count} duplicate rows (100% identical)")
                    
                    # Remove duplicates, keeping first occurrence
                    merged_df = merged_df.drop_duplicates(keep='first')
                    duplicates_removed = duplicate_count - len(merged_df) + original_rows
                    
                    logger.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Calculate merge statistics
            stats = {
                'merge_type': 'vertical_union',
                'original_rows_df1': len(df1),
                'original_rows_df2': len(df2),
                'merged_rows': len(merged_df),
                'columns_count': len(merged_df.columns),
                'duplicates_detected': duplicate_count if auto_drop_duplicates else 0,
                'duplicates_removed': duplicates_removed,
                'merge_timestamp': datetime.now().isoformat()
            }
            
            # Log merge operation
            self.merge_history.append(stats)
            
            logger.info(f"Vertical union merge completed: {len(merged_df)} rows final")
            
            return merged_df, stats
            
        except Exception as e:
            logger.error(f"Vertical union merge failed: {str(e)}")
            raise ValueError(f"Vertical union merge failed: {str(e)}")
    
    def smart_horizontal_join(self,
                            df1: pd.DataFrame,
                            df2: pd.DataFrame,
                            primary_key: str,
                            conflict_resolution: str = 'flag',
                            left_suffix: str = '_left',
                            right_suffix: str = '_right') -> Tuple[pd.DataFrame, Dict]:
        """
        Perform smart horizontal join with conflict detection and resolution
        
        Args:
            df1: Left DataFrame
            df2: Right DataFrame
            primary_key: Column name to join on
            conflict_resolution: How to handle conflicts ('flag', 'keep_left', 'keep_right', 'merge')
            left_suffix: Suffix for left DataFrame columns
            right_suffix: Suffix for right DataFrame columns
            
        Returns:
            Tuple of (joined DataFrame, join statistics)
        """
        
        try:
            logger.info(f"Starting smart horizontal join on '{primary_key}'")
            
            # Validate primary key exists in both DataFrames
            if primary_key not in df1.columns:
                raise ValueError(f"Primary key '{primary_key}' not found in left DataFrame")
            if primary_key not in df2.columns:
                raise ValueError(f"Primary key '{primary_key}' not found in right DataFrame")
            
            # Check for duplicate primary keys in individual DataFrames
            df1_duplicates = df1[primary_key].duplicated().sum()
            df2_duplicates = df2[primary_key].duplicated().sum()
            
            if df1_duplicates > 0:
                logger.warning(f"Left DataFrame has {df1_duplicates} duplicate primary keys")
            if df2_duplicates > 0:
                logger.warning(f"Right DataFrame has {df2_duplicates} duplicate primary keys")
            
            # Perform initial merge
            merged_df = pd.merge(df1, df2, on=primary_key, how='outer', 
                               suffixes=(left_suffix, right_suffix), indicator=True)
            
            # Analyze merge results
            left_only = merged_df['_merge'] == 'left_only'
            right_only = merged_df['_merge'] == 'right_only'
            both = merged_df['_merge'] == 'both'
            
            # Detect conflicts
            conflicts = []
            conflict_count = 0
            
            if conflict_resolution == 'flag' and both.sum() > 0:
                # Check for data conflicts in overlapping columns
                overlapping_columns = []
                
                for col in df1.columns:
                    if col != primary_key and col in df2.columns:
                        overlapping_columns.append(col)
                
                for idx, row in merged_df[both].iterrows():
                    pk_value = row[primary_key]
                    row_conflicts = []
                    
                    for col in overlapping_columns:
                        left_col = col + left_suffix
                        right_col = col + right_suffix
                        
                        if pd.notna(row[left_col]) and pd.notna(row[right_col]):
                            if row[left_col] != row[right_col]:
                                conflict_info = {
                                    'primary_key': primary_key,
                                    'pk_value': pk_value,
                                    'column': col,
                                    'left_value': row[left_col],
                                    'right_value': row[right_col]
                                }
                                conflicts.append(conflict_info)
                                row_conflicts.append(conflict_info)
                                conflict_count += 1
                    
                    if row_conflicts:
                        # Add conflict flag
                        merged_df.loc[idx, '_conflict_flag'] = True
                        merged_df.loc[idx, '_conflict_details'] = str(row_conflicts)
            
            # Apply conflict resolution strategy
            if conflict_resolution == 'keep_left' and both.sum() > 0:
                # Keep left values, drop right versions
                for col in df1.columns:
                    if col != primary_key and col in df2.columns:
                        right_col = col + right_suffix
                        if right_col in merged_df.columns:
                            merged_df.drop(columns=[right_col], inplace=True)
            
            elif conflict_resolution == 'keep_right' and both.sum() > 0:
                # Keep right values, drop left versions
                for col in df1.columns:
                    if col != primary_key and col in df2.columns:
                        left_col = col + left_suffix
                        if left_col in merged_df.columns:
                            merged_df.drop(columns=[left_col], inplace=True)
            
            elif conflict_resolution == 'merge' and both.sum() > 0:
                # Attempt to merge conflicting values
                for col in df1.columns:
                    if col != primary_key and col in df2.columns:
                        left_col = col + left_suffix
                        right_col = col + right_suffix
                        
                        if left_col in merged_df.columns and right_col in merged_df.columns:
                            # Create merged column
                            def merge_values(row):
                                if pd.isna(row[left_col]):
                                    return row[right_col]
                                elif pd.isna(row[right_col]):
                                    return row[left_col]
                                elif row[left_col] == row[right_col]:
                                    return row[left_col]
                                else:
                                    # Conflict - keep both values
                                    return f"{row[left_col]} | {row[right_col]}"
                            
                            merged_df[col] = merged_df.apply(merge_values, axis=1)
                            merged_df.drop(columns=[left_col, right_col], inplace=True)
            
            # Remove merge indicator column
            if '_merge' in merged_df.columns:
                merged_df.drop(columns=['_merge'], inplace=True)
            
            # Calculate join statistics
            stats = {
                'join_type': 'smart_horizontal',
                'primary_key': primary_key,
                'conflict_resolution': conflict_resolution,
                'left_rows': len(df1),
                'right_rows': len(df2),
                'joined_rows': len(merged_df),
                'left_only_count': left_only.sum(),
                'right_only_count': right_only.sum(),
                'both_count': both.sum(),
                'conflicts_detected': conflict_count,
                'conflict_details': conflicts[:10],  # Store first 10 conflicts
                'join_timestamp': datetime.now().isoformat()
            }
            
            # Log join operation
            self.merge_history.append(stats)
            
            # Store conflicts for UI display
            if conflicts:
                self.conflict_log.extend(conflicts)
                logger.warning(f"Detected {conflict_count} data conflicts during join")
            
            logger.info(f"Smart horizontal join completed: {len(merged_df)} rows final")
            
            return merged_df, stats
            
        except Exception as e:
            logger.error(f"Smart horizontal join failed: {str(e)}")
            raise ValueError(f"Smart horizontal join failed: {str(e)}")
    
    def vertical_concatenation(self,
                             dataframes: List[pd.DataFrame],
                             handle_mismatched_columns: str = 'union',
                             ignore_index: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform vertical concatenation (stacking) of multiple DataFrames
        
        Args:
            dataframes: List of DataFrames to concatenate
            handle_mismatched_columns: How to handle mismatched columns ('union', 'intersection', 'left')
            ignore_index: Whether to ignore the original index
            
        Returns:
            Tuple of (concatenated DataFrame, concatenation statistics)
        """
        
        try:
            logger.info(f"Starting vertical concatenation of {len(dataframes)} DataFrames")
            
            if not dataframes:
                raise ValueError("No DataFrames provided for concatenation")
            
            # Get column information
            all_columns = set()
            common_columns = set(dataframes[0].columns)
            df_columns = []
            
            for i, df in enumerate(dataframes):
                all_columns.update(df.columns)
                common_columns = common_columns.intersection(set(df.columns))
                df_columns.append(list(df.columns))
            
            # Handle mismatched columns
            if handle_mismatched_columns == 'union':
                # Use all columns from all DataFrames
                final_columns = list(all_columns)
                # Add missing columns with NaN values
                for i, df in enumerate(dataframes):
                    missing_cols = all_columns - set(df.columns)
                    for col in missing_cols:
                        dataframes[i][col] = pd.NA
                        
            elif handle_mismatched_columns == 'intersection':
                # Use only common columns
                final_columns = list(common_columns)
                for i, df in enumerate(dataframes):
                    dataframes[i] = df[final_columns]
                    
            elif handle_mismatched_columns == 'left':
                # Use columns from first DataFrame
                final_columns = list(dataframes[0].columns)
                for i, df in enumerate(dataframes[1:], 1):
                    missing_cols = set(final_columns) - set(df.columns)
                    for col in missing_cols:
                        dataframes[i][col] = pd.NA
                    extra_cols = set(df.columns) - set(final_columns)
                    dataframes[i] = dataframes[i].drop(columns=list(extra_cols))
            
            # Perform concatenation
            if ignore_index:
                concatenated_df = pd.concat(dataframes, ignore_index=True)
            else:
                concatenated_df = pd.concat(dataframes, ignore_index=False)
            
            # Calculate statistics
            total_original_rows = sum(len(df) for df in dataframes)
            
            stats = {
                'concatenation_type': 'vertical',
                'dataframe_count': len(dataframes),
                'original_rows_total': total_original_rows,
                'final_rows': len(concatenated_df),
                'final_columns': len(concatenated_df.columns),
                'handle_mismatched_columns': handle_mismatched_columns,
                'column_union_size': len(all_columns),
                'column_intersection_size': len(common_columns),
                'concatenation_timestamp': datetime.now().isoformat()
            }
            
            # Log operation
            self.merge_history.append(stats)
            
            logger.info(f"Vertical concatenation completed: {len(concatenated_df)} rows, {len(concatenated_df.columns)} columns")
            
            return concatenated_df, stats
            
        except Exception as e:
            logger.error(f"Vertical concatenation failed: {str(e)}")
            raise ValueError(f"Vertical concatenation failed: {str(e)}")
    
    def handle_missing_values(self,
                            df: pd.DataFrame,
                            strategy: str = 'drop',
                            columns: List[str] = None,
                            fill_value=None) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values in DataFrame
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_custom'
            columns: List of columns to process (None for all)
            fill_value: Custom fill value for 'fill_custom' strategy
            
        Returns:
            Tuple of (cleaned DataFrame, operation statistics)
        """
        
        try:
            logger.info(f"Starting missing value handling with strategy: {strategy}")
            
            if columns is None:
                columns = df.columns.tolist()
            
            original_rows = len(df)
            original_missing = df[columns].isnull().sum().sum()
            
            cleaned_df = df.copy()
            
            if strategy == 'drop':
                cleaned_df = cleaned_df.dropna(subset=columns)
                
            elif strategy == 'fill_mean':
                for col in columns:
                    if cleaned_df[col].dtype in ['int64', 'float64']:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                        
            elif strategy == 'fill_median':
                for col in columns:
                    if cleaned_df[col].dtype in ['int64', 'float64']:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                        
            elif strategy == 'fill_mode':
                for col in columns:
                    mode_value = cleaned_df[col].mode()
                    if not mode_value.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
                        
            elif strategy == 'fill_custom':
                if fill_value is not None:
                    cleaned_df[columns] = cleaned_df[columns].fillna(fill_value)
            
            final_rows = len(cleaned_df)
            final_missing = cleaned_df[columns].isnull().sum().sum()
            
            stats = {
                'strategy': strategy,
                'original_rows': original_rows,
                'final_rows': final_rows,
                'rows_removed': original_rows - final_rows,
                'original_missing': int(original_missing),
                'final_missing': int(final_missing),
                'missing_handled': int(original_missing - final_missing),
                'processed_columns': columns,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Missing value handling completed: {stats['rows_removed']} rows removed, {stats['missing_handled']} values filled")
            
            return cleaned_df, stats
            
        except Exception as e:
            logger.error(f"Missing value handling failed: {str(e)}")
            raise ValueError(f"Missing value handling failed: {str(e)}")
    
    def standardize_data_types(self,
                             df: pd.DataFrame,
                             numeric_columns: List[str] = None,
                             date_columns: List[str] = None,
                             date_format: str = 'YYYY-MM-DD') -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize data types in DataFrame
        
        Args:
            df: Input DataFrame
            numeric_columns: List of columns to convert to numeric
            date_columns: List of columns to convert to datetime
            date_format: Target date format
            
        Returns:
            Tuple of (standardized DataFrame, operation statistics)
        """
        
        try:
            logger.info("Starting data type standardization")
            
            standardized_df = df.copy()
            stats = {
                'numeric_conversions': {},
                'date_conversions': {},
                'errors': []
            }
            
            # Convert numeric columns
            if numeric_columns:
                for col in numeric_columns:
                    if col in standardized_df.columns:
                        try:
                            # Remove commas and convert to numeric
                            standardized_df[col] = standardized_df[col].astype(str).str.replace(',', '')
                            standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
                            stats['numeric_conversions'][col] = 'success'
                        except Exception as e:
                            stats['numeric_conversions'][col] = f'error: {str(e)}'
                            stats['errors'].append(f"Failed to convert {col} to numeric: {str(e)}")
            
            # Convert date columns
            if date_columns:
                for col in date_columns:
                    if col in standardized_df.columns:
                        try:
                            standardized_df[col] = pd.to_datetime(standardized_df[col], errors='coerce')
                            stats['date_conversions'][col] = 'success'
                        except Exception as e:
                            stats['date_conversions'][col] = f'error: {str(e)}'
                            stats['errors'].append(f"Failed to convert {col} to date: {str(e)}")
            
            stats['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Data type standardization completed: {len(stats['numeric_conversions'])} numeric, {len(stats['date_conversions'])} date conversions")
            
            return standardized_df, stats
            
        except Exception as e:
            logger.error(f"Data type standardization failed: {str(e)}")
            raise ValueError(f"Data type standardization failed: {str(e)}")
    
    def normalize_text_data(self,
                          df: pd.DataFrame,
                          text_columns: List[str] = None,
                          case_format: str = 'lower') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize text data in DataFrame
        
        Args:
            df: Input DataFrame
            text_columns: List of columns to normalize
            case_format: 'lower', 'upper', 'title', 'capitalize'
            
        Returns:
            Tuple of (normalized DataFrame, operation statistics)
        """
        
        try:
            logger.info(f"Starting text normalization with case format: {case_format}")
            
            if text_columns is None:
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            normalized_df = df.copy()
            stats = {
                'normalized_columns': [],
                'case_format': case_format,
                'errors': []
            }
            
            for col in text_columns:
                if col in normalized_df.columns:
                    try:
                        if case_format == 'lower':
                            normalized_df[col] = normalized_df[col].astype(str).str.lower()
                        elif case_format == 'upper':
                            normalized_df[col] = normalized_df[col].astype(str).str.upper()
                        elif case_format == 'title':
                            normalized_df[col] = normalized_df[col].astype(str).str.title()
                        elif case_format == 'capitalize':
                            normalized_df[col] = normalized_df[col].astype(str).str.capitalize()
                        
                        stats['normalized_columns'].append(col)
                    except Exception as e:
                        stats['errors'].append(f"Failed to normalize {col}: {str(e)}")
            
            stats['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Text normalization completed: {len(stats['normalized_columns'])} columns normalized")
            
            return normalized_df, stats
            
        except Exception as e:
            logger.error(f"Text normalization failed: {str(e)}")
            raise ValueError(f"Text normalization failed: {str(e)}")
    
    def restructure_columns(self,
                          df: pd.DataFrame,
                          operations: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
        """
        Restructure columns in DataFrame
        
        Args:
            df: Input DataFrame
            operations: List of operations to perform
                      Each operation is a dict with 'type' and parameters
                      Types: 'rename', 'delete', 'split', 'merge', 'reorder'
            
        Returns:
            Tuple of (restructured DataFrame, operation statistics)
        """
        
        try:
            logger.info(f"Starting column restructuring with {len(operations)} operations")
            
            restructured_df = df.copy()
            stats = {
                'operations': [],
                'errors': []
            }
            
            for i, operation in enumerate(operations):
                op_type = operation.get('type')
                
                try:
                    if op_type == 'rename':
                        mapping = operation.get('mapping', {})
                        restructured_df = restructured_df.rename(columns=mapping)
                        stats['operations'].append(f"Renamed {len(mapping)} columns")
                        
                    elif op_type == 'delete':
                        columns_to_delete = operation.get('columns', [])
                        restructured_df = restructured_df.drop(columns=columns_to_delete)
                        stats['operations'].append(f"Deleted {len(columns_to_delete)} columns")
                        
                    elif op_type == 'split':
                        column = operation.get('column')
                        new_columns = operation.get('new_columns', [])
                        separator = operation.get('separator', ' ')
                        
                        if column in restructured_df.columns:
                            split_data = restructured_df[column].astype(str).str.split(separator, expand=True)
                            for j, new_col in enumerate(new_columns):
                                if j < split_data.shape[1]:
                                    restructured_df[new_col] = split_data.iloc[:, j]
                            stats['operations'].append(f"Split {column} into {len(new_columns)} columns")
                            
                    elif op_type == 'merge':
                        columns_to_merge = operation.get('columns', [])
                        new_column = operation.get('new_column')
                        separator = operation.get('separator', ' ')
                        
                        if all(col in restructured_df.columns for col in columns_to_merge):
                            restructured_df[new_column] = restructured_df[columns_to_merge].astype(str).apply(
                                lambda x: separator.join(x), axis=1)
                            stats['operations'].append(f"Merged {len(columns_to_merge)} columns into {new_column}")
                            
                    elif op_type == 'reorder':
                        new_order = operation.get('order', [])
                        available_columns = [col for col in new_order if col in restructured_df.columns]
                        other_columns = [col for col in restructured_df.columns if col not in new_order]
                        restructured_df = restructured_df[available_columns + other_columns]
                        stats['operations'].append(f"Reordered {len(available_columns)} columns")
                        
                except Exception as e:
                    error_msg = f"Operation {i+1} ({op_type}) failed: {str(e)}"
                    stats['errors'].append(error_msg)
            
            stats['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Column restructuring completed: {len(stats['operations'])} successful, {len(stats['errors'])} errors")
            
            return restructured_df, stats
            
        except Exception as e:
            logger.error(f"Column restructuring failed: {str(e)}")
            raise ValueError(f"Column restructuring failed: {str(e)}")
    
    def heuristic_column_detection(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Use regex patterns to automatically identify and label column types
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to detected types
        """
        
        detected_types = {}
        
        for column in df.columns:
            col_data = df[column].dropna().head(100)  # Sample first 100 non-null values
            
            if len(col_data) == 0:
                detected_types[column] = 'empty'
                continue
            
            # Convert to string for pattern matching
            str_data = col_data.astype(str)
            
            # Test each pattern
            type_scores = {}
            
            for col_type, pattern in self.patterns.items():
                # Special handling for phone detection - check for number words first
                if col_type == 'phone':
                    # Check if column contains number words (if so, skip phone detection)
                    number_word_map = {
                        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
                        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
                        'eighteen': 18, 'nineteen': 19, 'twenty': 20
                    }
                    
                    has_number_words = 0
                    for val in col_data.head(20):
                        if pd.notna(val) and str(val).lower().strip() in number_word_map:
                            has_number_words += 1
                    
                    # If number words found, skip phone detection
                    if has_number_words > 0:
                        continue  # Skip to next pattern
                
                matches = str_data.str.match(pattern).sum()
                score = matches / len(str_data) if len(str_data) > 0 else 0
                type_scores[col_type] = score
            
            # Determine best match (score > 0.8 required)
            best_type = 'text'
            best_score = 0
            
            for col_type, score in type_scores.items():
                if score > best_score and score > 0.8:
                    best_type = col_type
                    best_score = score
            
            detected_types[column] = best_type
        
        logger.info(f"Column detection completed: {detected_types}")
        return detected_types
    
    def mask_pii_data(self, df: pd.DataFrame, 
                     column_types: Dict[str, str] = None,
                     masking_rules: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Mask Personally Identifiable Information (PII) for privacy
        
        Args:
            df: DataFrame to mask
            column_types: Dictionary of column types (if None, will auto-detect)
            masking_rules: Custom masking rules per column type
            
        Returns:
            Tuple of (masked DataFrame, masking statistics)
        """
        
        try:
            logger.info("Starting PII masking for privacy protection")
            
            # Auto-detect column types if not provided
            if column_types is None:
                column_types = self.heuristic_column_detection(df)
            
            # Default masking rules
            default_rules = {
                'email': lambda x: x[:2] + '***' + x.split('@')[1].split('.')[0][-2:] + '@' + x.split('@')[1],
                'phone': lambda x: '***' + str(x)[-4:] if len(str(x)) >= 4 else '***',
                'pincode': lambda x: '***' + str(x)[-3:] if len(str(x)) >= 3 else '***',
                'coordinates': lambda x: '***.***,***.***'
            }
            
            # Use custom rules if provided
            rules = masking_rules if masking_rules else default_rules
            
            masked_df = df.copy()
            masking_stats = {
                'total_columns': len(df.columns),
                'masked_columns': 0,
                'masking_details': {}
            }
            
            # Apply masking to identified PII columns
            for column, col_type in column_types.items():
                if col_type in rules and column in masked_df.columns:
                    original_values = masked_df[column].copy()
                    
                    # Apply masking function
                    try:
                        masked_df[column] = masked_df[column].apply(
                            lambda x: rules[col_type](x) if pd.notna(x) else x
                        )
                        
                        # Count masked values
                        masked_count = (original_values != masked_df[column]).sum()
                        
                        masking_stats['masked_columns'] += 1
                        masking_stats['masking_details'][column] = {
                            'type': col_type,
                            'masked_count': masked_count,
                            'total_count': len(original_values)
                        }
                        
                        logger.info(f"Masked {masked_count} values in column '{column}' ({col_type})")
                        
                    except Exception as e:
                        logger.warning(f"Failed to mask column '{column}': {str(e)}")
            
            masking_stats['masking_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"PII masking completed: {masking_stats['masked_columns']} columns masked")
            
            return masked_df, masking_stats
            
        except Exception as e:
            logger.error(f"PII masking failed: {str(e)}")
            raise ValueError(f"PII masking failed: {str(e)}")
    
    def anonymize_data(self, df: pd.DataFrame, 
                       columns_to_anonymize: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete anonymization of specified columns
        
        Args:
            df: DataFrame to anonymize
            columns_to_anonymize: List of columns to anonymize (if None, auto-detect PII)
            
        Returns:
            Tuple of (anonymized DataFrame, anonymization statistics)
        """
        
        try:
            logger.info("Starting complete data anonymization")
            
            # Auto-detect PII columns if not specified
            if columns_to_anonymize is None:
                column_types = self.heuristic_column_detection(df)
                pii_types = ['email', 'phone', 'pincode']
                columns_to_anonymize = [
                    col for col, col_type in column_types.items() 
                    if col_type in pii_types
                ]
            
            anonymized_df = df.copy()
            anon_stats = {
                'anonymized_columns': len(columns_to_anonymize),
                'anonymization_details': {}
            }
            
            for column in columns_to_anonymize:
                if column in anonymized_df.columns:
                    # Generate random identifiers
                    original_values = anonymized_df[column].copy()
                    unique_values = original_values.dropna().unique()
                    
                    # Create mapping for consistent anonymization
                    value_mapping = {}
                    for i, value in enumerate(unique_values):
                        value_mapping[value] = f"ANON_{i+1:06d}"
                    
                    # Apply anonymization
                    anonymized_df[column] = anonymized_df[column].map(value_mapping)
                    
                    anon_stats['anonymization_details'][column] = {
                        'unique_values': len(unique_values),
                        'anonymized_count': len(original_values.dropna())
                    }
                    
                    logger.info(f"Anonymized column '{column}' with {len(unique_values)} unique values")
            
            anon_stats['anonymization_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Data anonymization completed: {anon_stats['anonymized_columns']} columns")
            
            return anonymized_df, anon_stats
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {str(e)}")
            raise ValueError(f"Data anonymization failed: {str(e)}")
    
    def get_merge_history(self) -> List[Dict]:
        """Get history of all merge operations"""
        return self.merge_history.copy()
    
    def get_conflict_log(self) -> List[Dict]:
        """Get log of all detected conflicts"""
        return self.conflict_log.copy()
    
    def clear_history(self):
        """Clear merge history and conflict log"""
        self.merge_history.clear()
        self.conflict_log.clear()
        logger.info("Merge history and conflict log cleared")
    
    def validate_merge_operation(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               operation: str, **kwargs) -> Tuple[bool, List[str]]:
        """
        Validate merge operation before execution
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            operation: Type of operation
            **kwargs: Operation-specific parameters
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        
        errors = []
        
        try:
            if operation == 'vertical_union':
                # Check if DataFrames are too different
                if not df1.empty and not df2.empty:
                    common_cols = set(df1.columns) & set(df2.columns)
                    if len(common_cols) == 0:
                        errors.append("No common columns found between DataFrames")
            
            elif operation == 'smart_horizontal':
                primary_key = kwargs.get('primary_key')
                if not primary_key:
                    errors.append("Primary key is required for horizontal join")
                elif primary_key not in df1.columns:
                    errors.append(f"Primary key '{primary_key}' not found in left DataFrame")
                elif primary_key not in df2.columns:
                    errors.append(f"Primary key '{primary_key}' not found in right DataFrame")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
