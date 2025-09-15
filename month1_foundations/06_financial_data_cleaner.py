"""
Financial Data Cleaner - Systematic Data Cleaning for Financial Datasets
========================================================================

This module provides comprehensive data cleaning capabilities specifically
designed for financial datasets, following the principle of minimal intervention
while maintaining data integrity and providing complete audit trails.

Author: Extracted from Financial Engineering with AI and Blockchain textbook
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings

class FinancialDataCleaner:
    """
    Comprehensive data cleaning class for financial datasets
    
    This class implements systematic data cleaning procedures while
    maintaining data integrity and providing complete audit trails.
    """
    
    def __init__(self, data: pd.DataFrame, cleaning_config: Optional[Dict] = None):
        """
        Initialize the data cleaner
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to clean
        cleaning_config : dict, optional
            Configuration parameters for cleaning operations
        """
        self.original_data = data.copy()
        self.cleaned_data = data.copy()
        self.cleaning_config = cleaning_config or self._default_config()
        self.cleaning_log = []
        self.cleaning_stats = {}
    
    def _default_config(self) -> Dict:
        """Default configuration for cleaning operations"""
        return {
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'missing_threshold': 0.5,  # Drop columns with >50% missing
            'imputation_method': 'forward_fill',
            'business_rules': {
                'min_price': 0.01,  # Minimum valid price
                'max_return': 10.0,  # Maximum daily return (1000%)
                'min_volume': 0  # Minimum volume
            }
        }
    
    def clean_comprehensive(self) -> tuple:
        """
        Perform comprehensive data cleaning following best practices
        
        Returns:
        --------
        pd.DataFrame
            Cleaned dataset
        dict
            Cleaning statistics and log
        """
        print("COMPREHENSIVE DATA CLEANING PROCESS")
        print("=" * 50)
        
        # Step 1: Remove duplicate records
        self._remove_duplicates()
        
        # Step 2: Handle missing data
        self._handle_missing_data()
        
        # Step 3: Clean outliers
        self._clean_outliers()
        
        # Step 4: Apply business rules
        self._apply_business_rules()
        
        # Step 5: Standardize formats
        self._standardize_formats()
        
        # Step 6: Validate cleaning results
        self._validate_cleaning_results()
        
        # Generate final report
        self._generate_cleaning_report()
        
        return self.cleaned_data, self.cleaning_stats
    
    def _remove_duplicates(self):
        """Remove duplicate records with logging"""
        initial_rows = len(self.cleaned_data)
        
        # Identify duplicates
        duplicates = self.cleaned_data.duplicated()
        n_duplicates = duplicates.sum()
        
        if n_duplicates > 0:
            # Remove duplicates
            self.cleaned_data = self.cleaned_data.drop_duplicates()
            
            # Log the operation
            self.cleaning_log.append({
                'operation': 'remove_duplicates',
                'timestamp': datetime.now(),
                'records_removed': n_duplicates,
                'details': f"Removed {n_duplicates} duplicate records"
            })
            
            print(f"✓ Removed {n_duplicates} duplicate records")
        else:
            print("✓ No duplicate records found")
        
        final_rows = len(self.cleaned_data)
        self.cleaning_stats['duplicates_removed'] = initial_rows - final_rows
    
    def _handle_missing_data(self):
        """Handle missing data using configured strategy"""
        print(f"\nHANDLING MISSING DATA:")
        print("-" * 25)
        
        missing_summary = {}
        
        for column in self.cleaned_data.columns:
            missing_count = self.cleaned_data[column].isnull().sum()
            missing_percentage = (missing_count / len(self.cleaned_data)) * 100
            
            missing_summary[column] = {
                'count': missing_count,
                'percentage': missing_percentage
            }
            
            if missing_percentage > 0:
                print(f"{column}: {missing_count} missing ({missing_percentage:.1f}%)")
                
                # Apply cleaning strategy based on missing percentage
                if missing_percentage > self.cleaning_config['missing_threshold'] * 100:
                    # Drop column if too much missing data
                    self.cleaned_data = self.cleaned_data.drop(columns=[column])
                    
                    self.cleaning_log.append({
                        'operation': 'drop_column',
                        'timestamp': datetime.now(),
                        'column': column,
                        'reason': f"Missing data percentage ({missing_percentage:.1f}%) exceeds threshold"
                    })
                    
                    print(f"  → Dropped column (>{self.cleaning_config['missing_threshold']*100}% missing)")
                
                else:
                    # Apply imputation
                    self._impute_missing_values(column, missing_count)
        
        self.cleaning_stats['missing_data_summary'] = missing_summary
    
    def _impute_missing_values(self, column: str, missing_count: int):
        """Impute missing values based on data type and configuration"""
        
        if self.cleaned_data[column].dtype in ['float64', 'int64']:
            # Numeric data imputation
            if self.cleaning_config['imputation_method'] == 'forward_fill':
                self.cleaned_data[column] = self.cleaned_data[column].fillna(method='ffill')
                method_used = "forward fill"
            
            elif self.cleaning_config['imputation_method'] == 'mean':
                mean_value = self.cleaned_data[column].mean()
                self.cleaned_data[column] = self.cleaned_data[column].fillna(mean_value)
                method_used = f"mean ({mean_value:.4f})"
            
            elif self.cleaning_config['imputation_method'] == 'median':
                median_value = self.cleaned_data[column].median()
                self.cleaned_data[column] = self.cleaned_data[column].fillna(median_value)
                method_used = f"median ({median_value:.4f})"
            
            else:
                # Default to forward fill
                self.cleaned_data[column] = self.cleaned_data[column].fillna(method='ffill')
                method_used = "forward fill (default)"
        
        else:
            # Categorical data imputation
            mode_value = self.cleaned_data[column].mode()
            if len(mode_value) > 0:
                self.cleaned_data[column] = self.cleaned_data[column].fillna(mode_value[0])
                method_used = f"mode ({mode_value[0]})"
            else:
                self.cleaned_data[column] = self.cleaned_data[column].fillna('Unknown')
                method_used = "Unknown"
        
        # Log the imputation
        self.cleaning_log.append({
            'operation': 'impute_missing',
            'timestamp': datetime.now(),
            'column': column,
            'method': method_used,
            'values_imputed': missing_count
        })
        
        print(f"  → Imputed using {method_used}")
    
    def _clean_outliers(self):
        """Clean outliers using configured method"""
        print(f"\nCLEANING OUTLIERS:")
        print("-" * 18)
        
        numeric_columns = self.cleaned_data.select_dtypes(include=[np.number]).columns
        outliers_summary = {}
        
        for column in numeric_columns:
            series = self.cleaned_data[column].dropna()
            
            if len(series) > 10:  # Need sufficient data
                outliers_removed = self._clean_column_outliers(column, series)
                outliers_summary[column] = outliers_removed
                
                if outliers_removed > 0:
                    print(f"{column}: Cleaned {outliers_removed} outliers")
        
        self.cleaning_stats['outliers_summary'] = outliers_summary
    
    def _clean_column_outliers(self, column: str, series: pd.Series) -> int:
        """Clean outliers for a specific column"""
        
        if self.cleaning_config['outlier_method'] == 'iqr':
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            threshold = self.cleaning_config['outlier_threshold']
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Identify outliers
            outliers_mask = (self.cleaned_data[column] < lower_bound) | (self.cleaned_data[column] > upper_bound)
            
        elif self.cleaning_config['outlier_method'] == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            outliers_mask = z_scores > self.cleaning_config['outlier_threshold']
            
        else:
            return 0  # No cleaning performed
        
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            # Cap outliers instead of removing them (conservative approach)
            if self.cleaning_config['outlier_method'] == 'iqr':
                self.cleaned_data.loc[self.cleaned_data[column] < lower_bound, column] = lower_bound
                self.cleaned_data.loc[self.cleaned_data[column] > upper_bound, column] = upper_bound
            
            # Log the operation
            self.cleaning_log.append({
                'operation': 'clean_outliers',
                'timestamp': datetime.now(),
                'column': column,
                'method': self.cleaning_config['outlier_method'],
                'outliers_cleaned': outliers_count,
                'bounds': f"[{lower_bound:.4f}, {upper_bound:.4f}]" if self.cleaning_config['outlier_method'] == 'iqr' else None
            })
        
        return outliers_count
    
    def _apply_business_rules(self):
        """Apply business rules specific to financial data"""
        print(f"\nAPPLYING BUSINESS RULES:")
        print("-" * 25)
        
        rules_applied = []
        
        # Rule 1: Minimum price validation
        price_columns = [col for col in self.cleaned_data.columns 
                        if any(keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low'])]
        
        for column in price_columns:
            if self.cleaned_data[column].dtype in ['float64', 'int64']:
                min_price = self.cleaning_config['business_rules']['min_price']
                invalid_prices = (self.cleaned_data[column] < min_price).sum()
                
                if invalid_prices > 0:
                    # Set minimum price
                    self.cleaned_data.loc[self.cleaned_data[column] < min_price, column] = min_price
                    
                    rules_applied.append({
                        'rule': 'minimum_price',
                        'column': column,
                        'corrections': invalid_prices,
                        'threshold': min_price
                    })
                    
                    print(f"✓ {column}: Set {invalid_prices} values to minimum price ${min_price}")
        
        # Rule 2: Volume validation
        volume_columns = [col for col in self.cleaned_data.columns 
                         if any(keyword in col.lower() for keyword in ['volume', 'shares', 'quantity'])]
        
        for column in volume_columns:
            if self.cleaned_data[column].dtype in ['float64', 'int64']:
                min_volume = self.cleaning_config['business_rules']['min_volume']
                invalid_volumes = (self.cleaned_data[column] < min_volume).sum()
                
                if invalid_volumes > 0:
                    self.cleaned_data.loc[self.cleaned_data[column] < min_volume, column] = min_volume
                    
                    rules_applied.append({
                        'rule': 'minimum_volume',
                        'column': column,
                        'corrections': invalid_volumes,
                        'threshold': min_volume
                    })
                    
                    print(f"✓ {column}: Set {invalid_volumes} values to minimum volume {min_volume}")
        
        # Rule 3: Return validation
        return_columns = [col for col in self.cleaned_data.columns 
                         if any(keyword in col.lower() for keyword in ['return', 'yield', 'rate'])]
        
        for column in return_columns:
            if self.cleaned_data[column].dtype in ['float64', 'int64']:
                max_return = self.cleaning_config['business_rules']['max_return']
                extreme_returns = (np.abs(self.cleaned_data[column]) > max_return).sum()
                
                if extreme_returns > 0:
                    # Cap extreme returns
                    self.cleaned_data.loc[self.cleaned_data[column] > max_return, column] = max_return
                    self.cleaned_data.loc[self.cleaned_data[column] < -max_return, column] = -max_return
                    
                    rules_applied.append({
                        'rule': 'maximum_return',
                        'column': column,
                        'corrections': extreme_returns,
                        'threshold': max_return
                    })
                    
                    print(f"✓ {column}: Capped {extreme_returns} extreme returns at ±{max_return}")
        
        self.cleaning_stats['business_rules_applied'] = rules_applied
        
        # Log all business rule applications
        for rule in rules_applied:
            self.cleaning_log.append({
                'operation': 'apply_business_rule',
                'timestamp': datetime.now(),
                'rule': rule['rule'],
                'column': rule['column'],
                'corrections': rule['corrections']
            })
    
    def _standardize_formats(self):
        """Standardize data formats"""
        print(f"\nSTANDARDIZING FORMATS:")
        print("-" * 22)
        
        format_changes = []
        
        # Standardize string columns
        string_columns = self.cleaned_data.select_dtypes(include=['object']).columns
        
        for column in string_columns:
            # Convert to consistent case (uppercase for symbols)
            if any(keyword in column.lower() for keyword in ['symbol', 'ticker', 'code']):
                original_values = self.cleaned_data[column].copy()
                self.cleaned_data[column] = self.cleaned_data[column].str.upper()
                
                changes = (original_values != self.cleaned_data[column]).sum()
                if changes > 0:
                    format_changes.append({
                        'column': column,
                        'change': 'uppercase',
                        'values_changed': changes
                    })
                    print(f"✓ {column}: Converted {changes} values to uppercase")
        
        self.cleaning_stats['format_changes'] = format_changes
    
    def _validate_cleaning_results(self):
        """Validate that cleaning operations achieved their goals"""
        print(f"\nVALIDATING CLEANING RESULTS:")
        print("-" * 30)
        
        validation_results = {}
        
        # Check for remaining missing data
        remaining_missing = self.cleaned_data.isnull().sum().sum()
        validation_results['remaining_missing'] = remaining_missing
        print(f"✓ Remaining missing values: {remaining_missing}")
        
        # Check for remaining duplicates
        remaining_duplicates = self.cleaned_data.duplicated().sum()
        validation_results['remaining_duplicates'] = remaining_duplicates
        print(f"✓ Remaining duplicates: {remaining_duplicates}")
        
        # Check data integrity
        data_integrity_issues = []
        
        # Verify price relationships (High >= Low)
        if 'High' in self.cleaned_data.columns and 'Low' in self.cleaned_data.columns:
            invalid_relationships = (self.cleaned_data['High'] < self.cleaned_data['Low']).sum()
            if invalid_relationships > 0:
                data_integrity_issues.append(f"High < Low in {invalid_relationships} records")
        
        validation_results['data_integrity_issues'] = data_integrity_issues
        
        if data_integrity_issues:
            print("⚠️ Data integrity issues found:")
            for issue in data_integrity_issues:
                print(f"  - {issue}")
        else:
            print("✓ No data integrity issues found")
        
        self.cleaning_stats['validation_results'] = validation_results
    
    def _generate_cleaning_report(self):
        """Generate comprehensive cleaning report"""
        print(f"\n" + "="*60)
        print("DATA CLEANING SUMMARY REPORT")
        print("="*60)
        
        # Overall statistics
        original_shape = self.original_data.shape
        cleaned_shape = self.cleaned_data.shape
        
        print(f"\nDATA TRANSFORMATION:")
        print(f"Original dataset: {original_shape[0]:,} rows × {original_shape[1]} columns")
        print(f"Cleaned dataset:  {cleaned_shape[0]:,} rows × {cleaned_shape[1]} columns")
        print(f"Rows removed:     {original_shape[0] - cleaned_shape[0]:,}")
        print(f"Columns removed:  {original_shape[1] - cleaned_shape[1]}")
        
        # Cleaning operations summary
        print(f"\nCLEANING OPERATIONS PERFORMED:")
        operation_counts = {}
        for log_entry in self.cleaning_log:
            op = log_entry['operation']
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        for operation, count in operation_counts.items():
            print(f"  {operation.replace('_', ' ').title()}: {count}")
        
        # Quality improvement metrics
        original_missing = self.original_data.isnull().sum().sum()
        cleaned_missing = self.cleaned_data.isnull().sum().sum()
        
        print(f"\nQUALITY IMPROVEMENTS:")
        print(f"Missing values: {original_missing:,} → {cleaned_missing:,} "
              f"({((original_missing - cleaned_missing) / original_missing * 100):.1f}% reduction)")
        
        if 'duplicates_removed' in self.cleaning_stats:
            print(f"Duplicates removed: {self.cleaning_stats['duplicates_removed']}")
        
        # Final recommendations
        print(f"\nRECOMMENDATIONS:")
        if cleaned_missing > 0:
            print(f"  • Review remaining {cleaned_missing} missing values")
        
        if len(self.cleaning_stats.get('validation_results', {}).get('data_integrity_issues', [])) > 0:
            print(f"  • Address data integrity issues identified")
        
        print(f"  • Implement automated data validation pipeline")
        print(f"  • Regular monitoring of data quality metrics")
        
        # Store final statistics
        self.cleaning_stats['final_summary'] = {
            'original_shape': original_shape,
            'cleaned_shape': cleaned_shape,
            'rows_removed': original_shape[0] - cleaned_shape[0],
            'columns_removed': original_shape[1] - cleaned_shape[1],
            'missing_reduction': original_missing - cleaned_missing,
            'operations_performed': len(self.cleaning_log)
        }

def main():
    """Main function to demonstrate the financial data cleaner"""
    
    # Create sample data with issues (reusing the profiler's function)
    from financial_data_profiler import create_realistic_financial_dataset
    
    print("Creating sample financial dataset with quality issues...")
    sample_data = create_realistic_financial_dataset()
    
    # Initialize cleaner
    cleaner = FinancialDataCleaner(sample_data)
    
    # Perform comprehensive cleaning
    cleaned_data, cleaning_stats = cleaner.clean_comprehensive()
    
    return cleaned_data, cleaning_stats

if __name__ == "__main__":
    main()

