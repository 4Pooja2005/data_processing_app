"""
Natural Language Command Parser Module
Processes user commands in natural language for data manipulation
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime

class NLPCommandParser:
    """
    Natural Language Command Parser for data manipulation
    Converts user-friendly commands into data operations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.command_history = []
        
        # Define command patterns
        self.patterns = {
            # Filter commands
            'filter_rows': [
                r'filter rows? where (.+)',
                r'show rows? where (.+)',
                'select rows? where (.+)',
                r'keep rows? where (.+)'
            ],
            'filter_columns': [
                r'select columns? (.+)',
                r'show columns? (.+)',
                r'keep columns? (.+)',
                r'filter columns? (.+)'
            ],
            
            # Data transformation commands
            'split_column': [
                r'split (.+) into (.+)',
                r'divide (.+) into (.+)',
                r'separate (.+) into (.+)'
            ],
            'combine_columns': [
                r'combine (.+) and (.+) into (.+)',
                r'merge (.+) and (.+) into (.+)',
                r'concatenate (.+) and (.+) into (.+)'
            ],
            'rename_column': [
                r'rename (.+) to (.+)',
                r'change (.+) to (.+)',
                r'set (.+) to (.+)'
            ],
            'delete_column': [
                r'delete (.+)',
                r'remove (.+)',
                r'drop (.+)'
            ],
            
            # Data cleaning commands
            'fill_missing': [
                r'fill missing (.+) with (.+)',
                r'replace missing (.+) with (.+)',
                r'impute (.+) with (.+)'
            ],
            'remove_duplicates': [
                r'remove duplicates',
                r'drop duplicates',
                r'eliminate duplicates'
            ],
            
            # Data type commands
            'convert_type': [
                r'convert (.+) to (.+)',
                r'change (.+) to (.+)',
                r'type (.+) as (.+)'
            ],
            
            # Sorting commands
            'sort_by': [
                r'sort by (.+)',
                r'order by (.+)',
                r'arrange by (.+)'
            ],
            
            # Aggregation commands
            'group_by': [
                r'group by (.+)',
                r'summarize by (.+)'
            ],
            
            # Export commands
            'export': [
                r'export (.+)',
                r'save (.+)',
                r'download (.+)'
            ]
        }
        
        # Define column type patterns
        self.column_type_patterns = {
            'email': [
                r'email', r'e-mail', r'electronic.?mail',
                r'.*email.*', r'.*mail.*'
            ],
            'phone': [
                r'phone', r'telephone', r'mobile', r'cell',
                r'.*phone.*', r'.*tel.*', r'.*mobile.*'
            ],
            'pincode': [
                r'pincode', r'pin', r'zipcode', r'zip', r'postcode',
                r'.*pin.*', r'.*zip.*', r'.*postal.*'
            ],
            'currency': [
                r'price', r'cost', r'amount', r'currency', r'money',
                r'.*price.*', r'.*cost.*', r'.*amount.*'
            ],
            'coordinates': [
                r'coordinate', r'location', r'gps', r'latitude', r'longitude',
                r'.*coord.*', r'.*location.*', r'.*gps.*'
            ],
            'date': [
                r'date', r'time', r'datetime', r'created', r'modified',
                r'.*date.*', r'.*time.*'
            ],
            'name': [
                r'name', r'full.?name', r'first.?name', r'last.?name',
                r'.*name.*'
            ],
            'id': [
                r'id', r'identifier', r'key', r'code',
                r'.*id.*', r'.*identifier.*'
            ]
        }
    
    def parse_command(self, command: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse natural language command and return operation details
        
        Args:
            command: User command string
            df: DataFrame to operate on
            
        Returns:
            Dictionary with parsed operation details
        """
        
        command = command.strip().lower()
        
        # Try to match command patterns
        for operation_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    result = self._parse_specific_command(operation_type, match, command, df)
                    if result:
                        # Log command
                        self.command_history.append({
                            'timestamp': datetime.now(),
                            'command': command,
                            'operation_type': operation_type,
                            'result': result
                        })
                        return result
        
        # If no pattern matched, return error
        return {
            'success': False,
            'error': f'Command not recognized: {command}',
            'suggestions': self._get_command_suggestions(command)
        }
    
    def _parse_specific_command(self, operation_type: str, match: re.Match, 
                              command: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse specific command types"""
        
        try:
            if operation_type == 'filter_rows':
                return self._parse_filter_rows(match, df)
            elif operation_type == 'filter_columns':
                return self._parse_filter_columns(match, df)
            elif operation_type == 'split_column':
                return self._parse_split_column(match, df)
            elif operation_type == 'combine_columns':
                return self._parse_combine_columns(match, df)
            elif operation_type == 'rename_column':
                return self._parse_rename_column(match, df)
            elif operation_type == 'delete_column':
                return self._parse_delete_column(match, df)
            elif operation_type == 'fill_missing':
                return self._parse_fill_missing(match, df)
            elif operation_type == 'remove_duplicates':
                return self._parse_remove_duplicates(df)
            elif operation_type == 'convert_type':
                return self._parse_convert_type(match, df)
            elif operation_type == 'sort_by':
                return self._parse_sort_by(match, df)
            elif operation_type == 'group_by':
                return self._parse_group_by(match, df)
            elif operation_type == 'export':
                return self._parse_export(match, df)
            else:
                return {'success': False, 'error': f'Unknown operation: {operation_type}'}
        
        except Exception as e:
            self.logger.error(f"Error parsing command {operation_type}: {str(e)}")
            return {'success': False, 'error': f'Error parsing command: {str(e)}'}
    
    def _parse_filter_rows(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse row filtering commands"""
        condition = match.group(1).strip()
        
        # Parse the condition
        filter_result = self._parse_filter_condition(condition, df)
        
        if not filter_result['success']:
            return filter_result
        
        return {
            'success': True,
            'operation': 'filter_rows',
            'condition': condition,
            'parsed_condition': filter_result['parsed'],
            'description': f'Filter rows where {condition}'
        }
    
    def _parse_filter_condition(self, condition: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse filter condition into executable format"""
        
        # Handle basic comparisons: column > value, column = value, etc.
        comparison_patterns = [
            r'(\w+)\s*(>=|<=|!=|=|>|<)\s*(.+)',
            r'(\w+)\s*(equals?|contains?|starts with|ends with)\s*(.+)'
        ]
        
        for pattern in comparison_patterns:
            match = re.search(pattern, condition, re.IGNORECASE)
            if match:
                column = match.group(1)
                operator = match.group(2)
                value = match.group(3).strip('\'"')
                
                # Check if column exists (case insensitive)
                df_columns_lower = {col.lower(): col for col in df.columns}
                column_lower = column.lower()
                if column_lower not in df_columns_lower:
                    return {'success': False, 'error': f'Column "{column}" not found'}
                
                # Use the actual column name from DataFrame
                actual_column = df_columns_lower[column_lower]
                
                # Parse operator
                if operator in ['=', 'equals']:
                    parsed_op = 'eq'
                elif operator in ['!=']:
                    parsed_op = 'ne'
                elif operator in ['>']:
                    parsed_op = 'gt'
                elif operator in ['<']:
                    parsed_op = 'lt'
                elif operator in ['>=']:
                    parsed_op = 'ge'
                elif operator in ['<=']:
                    parsed_op = 'le'
                elif operator in ['contains']:
                    parsed_op = 'contains'
                elif operator in ['starts with']:
                    parsed_op = 'startswith'
                elif operator in ['ends with']:
                    parsed_op = 'endswith'
                else:
                    return {'success': False, 'error': f'Unknown operator: {operator}'}
                
                # Try to convert value to appropriate type
                converted_value = self._convert_value(value, df[actual_column])
                
                return {
                    'success': True,
                    'parsed': {
                        'column': actual_column,
                        'operator': parsed_op,
                        'value': converted_value,
                        'original_value': value
                    }
                }
        
        # Handle logical conditions (AND, OR)
        if ' and ' in condition.lower():
            parts = condition.lower().split(' and ')
            parsed_parts = []
            for part in parts:
                part_result = self._parse_filter_condition(part.strip(), df)
                if not part_result['success']:
                    return part_result
                parsed_parts.append(part_result['parsed'])
            
            return {
                'success': True,
                'parsed': {
                    'logic': 'and',
                    'conditions': parsed_parts
                }
            }
        
        if ' or ' in condition.lower():
            parts = condition.lower().split(' or ')
            parsed_parts = []
            for part in parts:
                part_result = self._parse_filter_condition(part.strip(), df)
                if not part_result['success']:
                    return part_result
                parsed_parts.append(part_result['parsed'])
            
            return {
                'success': True,
                'parsed': {
                    'logic': 'or',
                    'conditions': parsed_parts
                }
            }
        
        return {'success': False, 'error': f'Cannot parse condition: {condition}'}
    
    def _convert_value(self, value: str, series: pd.Series) -> Any:
        """Convert string value to appropriate type based on series data type"""
        
        # Try numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try datetime conversion
        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.to_datetime(value)
            except:
                pass
        
        # Return as string
        return value
    
    def _parse_filter_columns(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse column filtering commands"""
        columns_spec = match.group(1).strip()
        
        # Create case-insensitive column mapping
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        # Parse column specification
        if columns_spec == 'all':
            columns = list(df.columns)
        elif columns_spec.startswith('except'):
            # "except column1, column2"
            except_cols = [col.strip().lower() for col in columns_spec[6:].split(',')]
            columns = [df_columns_lower[col] for col in df.columns if col.lower() not in except_cols]
        elif ',' in columns_spec:
            # "column1, column2, column3"
            cols_lower = [col.strip().lower() for col in columns_spec.split(',')]
            missing_cols = [col for col in cols_lower if col not in df_columns_lower]
            if missing_cols:
                return {'success': False, 'error': f'Columns not found: {missing_cols}'}
            columns = [df_columns_lower[col] for col in cols_lower]
        else:
            # Single column
            col_lower = columns_spec.lower()
            if col_lower not in df_columns_lower:
                return {'success': False, 'error': f'Column "{columns_spec}" not found'}
            columns = [df_columns_lower[col_lower]]
        
        return {
            'success': True,
            'operation': 'filter_columns',
            'columns': columns,
            'description': f'Select columns: {", ".join(columns)}'
        }
    
    def _parse_split_column(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse column splitting commands"""
        source_col = match.group(1).strip()
        target_cols = [col.strip() for col in match.group(2).split(',')]
        
        # Create case-insensitive column mapping
        df_columns_lower = {col.lower(): col for col in df.columns}
        source_col_lower = source_col.lower()
        
        if source_col_lower not in df_columns_lower:
            return {'success': False, 'error': f'Source column "{source_col}" not found'}
        
        actual_source_col = df_columns_lower[source_col_lower]
        
        return {
            'success': True,
            'operation': 'split_column',
            'source_column': actual_source_col,
            'target_columns': target_cols,
            'description': f'Split {actual_source_col} into {", ".join(target_cols)}'
        }
    
    def _parse_combine_columns(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse column combining commands"""
        col1 = match.group(1).strip()
        col2 = match.group(2).strip()
        target_col = match.group(3).strip()
        
        missing_cols = [col for col in [col1, col2] if col not in df.columns]
        if missing_cols:
            return {'success': False, 'error': f'Columns not found: {missing_cols}'}
        
        return {
            'success': True,
            'operation': 'combine_columns',
            'source_columns': [col1, col2],
            'target_column': target_col,
            'description': f'Combine {col1} and {col2} into {target_col}'
        }
    
    def _parse_rename_column(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse column renaming commands"""
        old_name = match.group(1).strip()
        new_name = match.group(2).strip()
        
        if old_name not in df.columns:
            return {'success': False, 'error': f'Column "{old_name}" not found'}
        
        return {
            'success': True,
            'operation': 'rename_column',
            'old_name': old_name,
            'new_name': new_name,
            'description': f'Rename {old_name} to {new_name}'
        }
    
    def _parse_delete_column(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse column deletion commands"""
        columns_spec = match.group(1).strip()
        
        if ',' in columns_spec:
            columns = [col.strip() for col in columns_spec.split(',')]
        else:
            columns = [columns_spec]
        
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return {'success': False, 'error': f'Columns not found: {missing_cols}'}
        
        return {
            'success': True,
            'operation': 'delete_column',
            'columns': columns,
            'description': f'Delete columns: {", ".join(columns)}'
        }
    
    def _parse_fill_missing(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse missing value filling commands"""
        column = match.group(1).strip()
        value = match.group(2).strip().strip('\'"')
        
        if column not in df.columns:
            return {'success': False, 'error': f'Column "{column}" not found'}
        
        # Convert value to appropriate type
        converted_value = self._convert_value(value, df[column])
        
        return {
            'success': True,
            'operation': 'fill_missing',
            'column': column,
            'value': converted_value,
            'description': f'Fill missing values in {column} with {value}'
        }
    
    def _parse_remove_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse duplicate removal commands"""
        return {
            'success': True,
            'operation': 'remove_duplicates',
            'description': 'Remove duplicate rows'
        }
    
    def _parse_convert_type(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse data type conversion commands"""
        column = match.group(1).strip()
        target_type = match.group(2).strip()
        
        if column not in df.columns:
            return {'success': False, 'error': f'Column "{column}" not found'}
        
        # Map common type names
        type_mapping = {
            'string': 'object',
            'text': 'object',
            'number': 'float64',
            'numeric': 'float64',
            'integer': 'int64',
            'int': 'int64',
            'float': 'float64',
            'datetime': 'datetime64[ns]',
            'date': 'datetime64[ns]',
            'boolean': 'bool',
            'bool': 'bool'
        }
        
        mapped_type = type_mapping.get(target_type.lower(), target_type)
        
        return {
            'success': True,
            'operation': 'convert_type',
            'column': column,
            'target_type': mapped_type,
            'description': f'Convert {column} to {target_type}'
        }
    
    def _parse_sort_by(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse sorting commands"""
        columns_spec = match.group(1).strip()
        
        # Parse columns and sort direction
        sort_columns = []
        ascending = []
        
        for col_spec in columns_spec.split(','):
            col_spec = col_spec.strip()
            if col_spec.lower().endswith(' desc'):
                sort_columns.append(col_spec[:-4].strip())
                ascending.append(False)
            elif col_spec.lower().endswith(' asc'):
                sort_columns.append(col_spec[:-3].strip())
                ascending.append(True)
            else:
                sort_columns.append(col_spec)
                ascending.append(True)
        
        missing_cols = [col for col in sort_columns if col not in df.columns]
        if missing_cols:
            return {'success': False, 'error': f'Columns not found: {missing_cols}'}
        
        return {
            'success': True,
            'operation': 'sort_by',
            'columns': sort_columns,
            'ascending': ascending,
            'description': f'Sort by {", ".join(sort_columns)}'
        }
    
    def _parse_group_by(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse grouping commands"""
        columns_spec = match.group(1).strip()
        
        if ',' in columns_spec:
            columns = [col.strip() for col in columns_spec.split(',')]
        else:
            columns = [columns_spec]
        
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return {'success': False, 'error': f'Columns not found: {missing_cols}'}
        
        return {
            'success': True,
            'operation': 'group_by',
            'columns': columns,
            'description': f'Group by {", ".join(columns)}'
        }
    
    def _parse_export(self, match: re.Match, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse export commands"""
        format_spec = match.group(1).strip()
        
        format_mapping = {
            'csv': 'csv',
            'excel': 'excel',
            'xlsx': 'excel',
            'json': 'json'
        }
        
        export_format = format_mapping.get(format_spec.lower(), 'csv')
        
        return {
            'success': True,
            'operation': 'export',
            'format': export_format,
            'description': f'Export data as {export_format}'
        }
    
    def _get_command_suggestions(self, command: str) -> List[str]:
        """Get command suggestions based on partial input"""
        suggestions = []
        
        # Simple keyword-based suggestions
        if 'filter' in command.lower():
            suggestions.extend([
                'filter rows where column > value',
                'filter columns column1, column2',
                'filter rows where column = "value"'
            ])
        elif 'split' in command.lower():
            suggestions.extend([
                'split column into col1, col2',
                'split name into first_name, last_name'
            ])
        elif 'combine' in command.lower():
            suggestions.extend([
                'combine col1 and col2 into new_col',
                'combine first_name and last_name into full_name'
            ])
        elif 'rename' in command.lower():
            suggestions.extend([
                'rename old_name to new_name',
                'rename column to new_column'
            ])
        elif 'delete' in command.lower() or 'remove' in command.lower():
            suggestions.extend([
                'delete column1',
                'remove column1, column2'
            ])
        elif 'sort' in command.lower():
            suggestions.extend([
                'sort by column',
                'sort by column1, column2 desc'
            ])
        
        return suggestions
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect column types using regex and heuristics
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to detected types
        """
        
        detected_types = {}
        
        for column in df.columns:
            col_name = str(column).lower()
            col_data = df[column].dropna().head(100)  # Sample first 100 non-null values
            
            # Check each type pattern
            for col_type, patterns in self.column_type_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, col_name, re.IGNORECASE):
                        detected_types[column] = col_type
                        break
                
                if column in detected_types:
                    break
            
            # If no pattern matched, try content-based detection
            if column not in detected_types:
                content_type = self._detect_content_type(col_data)
                if content_type:
                    detected_types[column] = content_type
                else:
                    detected_types[column] = 'text'
        
        return detected_types
    
    def _detect_content_type(self, series: pd.Series) -> Optional[str]:
        """Detect column type based on content"""
        
        if len(series) == 0:
            return None
        
        # Convert to string for pattern matching
        str_data = series.astype(str)
        
        # Email detection
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        email_matches = str_data.str.match(email_pattern).sum()
        if email_matches / len(str_data) > 0.8:
            return 'email'
        
        # Phone detection - but first check if this might be a number-word column
        # Import number word map for checking
        number_word_map = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }
        
        # Check if column contains number words (if so, don't detect as phone)
        has_number_words = 0
        for val in series.head(20):
            if pd.notna(val) and str(val).lower().strip() in number_word_map:
                has_number_words += 1
        
        # If significant number words found, don't detect as phone
        if has_number_words > 0:
            return None  # Let it fall through to text detection
        
        phone_pattern = r'^[\+]?[1-9][\d]{0,15}$|^\(?[\d]{3}\)?[-\s\.]?[\d]{3}[-\s\.]?[\d]{4}$'
        phone_matches = str_data.str.match(phone_pattern).sum()
        if phone_matches / len(str_data) > 0.8:
            return 'phone'
        
        # Pincode detection
        pincode_pattern = r'^\d{5}(-\d{4})?$|^\d{6}$|^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$'
        pincode_matches = str_data.str.match(pincode_pattern).sum()
        if pincode_matches / len(str_data) > 0.8:
            return 'pincode'
        
        # Currency detection
        currency_pattern = r'^\$?\s*[\d,]+\.?\d*$'
        currency_matches = str_data.str.match(currency_pattern).sum()
        if currency_matches / len(str_data) > 0.8:
            return 'currency'
        
        # Coordinates detection
        coord_pattern = r'^-?\d+\.?\d*,\s*-?\d+\.?\d*$'
        coord_matches = str_data.str.match(coord_pattern).sum()
        if coord_matches / len(str_data) > 0.8:
            return 'coordinates'
        
        return None
    
    def get_command_history(self) -> List[Dict]:
        """Get history of parsed commands"""
        return self.command_history.copy()
    
    def clear_command_history(self):
        """Clear command history"""
        self.command_history.clear()
    
    def get_supported_commands(self) -> Dict[str, List[str]]:
        """Get list of supported command patterns"""
        return self.patterns.copy()
