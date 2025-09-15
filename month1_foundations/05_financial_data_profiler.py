"""
Financial Data Profiler - Comprehensive Data Quality Analysis
============================================================

This module provides comprehensive data profiling capabilities specifically
designed for financial datasets, including quality assessment, statistical
analysis, and business rule validation.

Author: Extracted from Financial Engineering with AI and Blockchain textbook
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
from datetime import datetime
from typing import Dict, List, Any

class FinancialDataProfiler:
    """
    Comprehensive data profiling class for financial datasets
    
    This class analyzes data quality, statistical properties, and business rule
    compliance for financial data, providing detailed reports and recommendations
    for data cleaning and preprocessing.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the data profiler
        
        Parameters:
        -----------
        data : pd.DataFrame
            The financial dataset to profile
        """
        self.data = data.copy()
        self.profile_results = {}
        self.quality_issues = []
        
    def generate_comprehensive_profile(self) -> Dict[str, Any]:
        """
        Generate a comprehensive data profile
        
        Returns:
        --------
        dict
            Complete profiling results including quality metrics,
            statistical summaries, and recommendations
        """
        print("FINANCIAL DATA PROFILING - COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()
        
        # Run all profiling components
        self._profile_basic_info()
        self._profile_missing_data()
        self._profile_data_types()
        self._profile_distributions()
        self._profile_outliers()
        self._profile_consistency()
        self._profile_business_rules()
        self._generate_summary_report()
        
        return self.profile_results
    
    def _profile_basic_info(self):
        """Analyze basic dataset information"""
        basic_info = {
            'n_rows': len(self.data),
            'n_columns': len(self.data.columns),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': self.data.duplicated().sum(),
            'column_names': list(self.data.columns)
        }
        
        self.profile_results['basic_info'] = basic_info
        
        print("BASIC DATASET INFORMATION:")
        print("-" * 30)
        print(f"Rows: {basic_info['n_rows']:,}")
        print(f"Columns: {basic_info['n_columns']}")
        print(f"Memory usage: {basic_info['memory_usage_mb']:.2f} MB")
        print(f"Duplicate rows: {basic_info['duplicate_rows']}")
        
        if basic_info['duplicate_rows'] > 0:
            self.quality_issues.append(f"Dataset contains {basic_info['duplicate_rows']} duplicate rows")
    
    def _profile_missing_data(self):
        """Analyze missing data patterns"""
        missing_summary = []
        
        for column in self.data.columns:
            missing_count = self.data[column].isnull().sum()
            missing_percentage = (missing_count / len(self.data)) * 100
            
            missing_summary.append({
                'Column': column,
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_percentage
            })
        
        missing_df = pd.DataFrame(missing_summary)
        self.profile_results['missing_data'] = missing_df
        
        print(f"\nMISSING DATA ANALYSIS:")
        print("-" * 25)
        
        # Display columns with missing data
        columns_with_missing = missing_df[missing_df['Missing_Count'] > 0]
        if len(columns_with_missing) > 0:
            print("Columns with missing data:")
            for _, row in columns_with_missing.iterrows():
                print(f"  {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)")
                
                # Flag significant missing data
                if row['Missing_Percentage'] > 10:
                    self.quality_issues.append(f"Column '{row['Column']}' has {row['Missing_Percentage']:.1f}% missing data")
        else:
            print("No missing data found")
        
        # Calculate overall completeness
        total_cells = len(self.data) * len(self.data.columns)
        missing_cells = missing_df['Missing_Count'].sum()
        completeness_rate = ((total_cells - missing_cells) / total_cells) * 100
        
        self.profile_results['completeness'] = {
            'Completeness_Rate': completeness_rate,
            'Total_Missing_Cells': missing_cells
        }
        
        print(f"Overall completeness: {completeness_rate:.1f}%")
        
        # Flag low completeness
        if completeness_rate < 95:
            self.quality_issues.append(f"Overall data completeness is low ({completeness_rate:.1f}%)")
        
        # Missing data patterns
        missing_patterns = self.data.isnull().value_counts()
        if len(missing_patterns) > 1:
            print(f"\nMissing Data Patterns: {len(missing_patterns)} unique patterns found")
        
        # Consecutive missing values (important for time series)
        if isinstance(self.data.index, pd.DatetimeIndex):
            self._analyze_consecutive_missing()
    
    def _analyze_consecutive_missing(self):
        """Analyze consecutive missing values in time series data"""
        for column in self.data.columns:
            if self.data[column].dtype in ['float64', 'int64']:
                # Find consecutive missing value sequences
                is_missing = self.data[column].isnull()
                missing_groups = (is_missing != is_missing.shift()).cumsum()
                consecutive_missing = is_missing.groupby(missing_groups).sum()
                max_consecutive = consecutive_missing.max()
                
                if max_consecutive > 5:  # More than 5 consecutive missing values
                    self.quality_issues.append(
                        f"Column '{column}' has up to {max_consecutive} consecutive missing values"
                    )
    
    def _profile_data_types(self):
        """Analyze data types and formats"""
        type_summary = self.data.dtypes.value_counts()
        
        self.profile_results['data_types'] = {
            'type_distribution': type_summary.to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(self.data.select_dtypes(include=['datetime64']).columns)
        }
        
        print(f"\nDATA TYPE ANALYSIS:")
        for dtype, count in type_summary.items():
            print(f"  {dtype}: {count} columns")
        
        # Check for potential type issues
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                # Check if object column contains mostly numbers
                non_null_values = self.data[column].dropna()
                if len(non_null_values) > 0:
                    try:
                        pd.to_numeric(non_null_values.head(100))
                        self.quality_issues.append(
                            f"Column '{column}' is object type but appears to contain numeric data"
                        )
                    except (ValueError, TypeError):
                        pass
    
    def _profile_distributions(self):
        """Analyze statistical distributions of numeric columns"""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        distribution_stats = {}
        
        for column in numeric_columns:
            series = self.data[column].dropna()
            if len(series) > 0:
                stats_dict = {
                    'count': len(series),
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'skewness': stats.skew(series),
                    'kurtosis': stats.kurtosis(series),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75)
                }
                
                # Test for normality
                if len(series) >= 8:  # Minimum sample size for Shapiro-Wilk
                    try:
                        _, p_value = stats.shapiro(series.sample(min(5000, len(series))))
                        stats_dict['normality_p_value'] = p_value
                        stats_dict['is_normal'] = p_value > 0.05
                    except:
                        stats_dict['normality_p_value'] = None
                        stats_dict['is_normal'] = None
                
                distribution_stats[column] = stats_dict
        
        self.profile_results['distributions'] = distribution_stats
        
        print(f"\nDISTRIBUTION ANALYSIS:")
        for column, stats_dict in distribution_stats.items():
            print(f"\n{column}:")
            print(f"  Mean: {stats_dict['mean']:.4f}, Median: {stats_dict['median']:.4f}")
            print(f"  Std: {stats_dict['std']:.4f}, Skewness: {stats_dict['skewness']:.4f}")
            
            # Flag potential issues
            if abs(stats_dict['skewness']) > 2:
                self.quality_issues.append(f"Column '{column}' is highly skewed ({stats_dict['skewness']:.2f})")
            
            if stats_dict['kurtosis'] > 7:
                self.quality_issues.append(f"Column '{column}' has high kurtosis ({stats_dict['kurtosis']:.2f})")
    
    def _profile_outliers(self):
        """Detect and analyze outliers using multiple methods"""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for column in numeric_columns:
            series = self.data[column].dropna()
            if len(series) > 10:  # Need sufficient data for outlier detection
                outliers_dict = {}
                
                # Z-score method
                z_scores = np.abs(stats.zscore(series))
                z_outliers = (z_scores > 3).sum()
                outliers_dict['z_score_outliers'] = z_outliers
                
                # IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                outliers_dict['iqr_outliers'] = iqr_outliers
                
                # Modified Z-score method
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad
                modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
                outliers_dict['modified_z_outliers'] = modified_z_outliers
                
                outlier_summary[column] = outliers_dict
                
                # Flag significant outlier presence
                outlier_percentage = (iqr_outliers / len(series)) * 100
                if outlier_percentage > 5:
                    self.quality_issues.append(
                        f"Column '{column}' has {outlier_percentage:.1f}% outliers (IQR method)"
                    )
        
        self.profile_results['outliers'] = outlier_summary
        
        print(f"\nOUTLIER ANALYSIS:")
        for column, outliers_dict in outlier_summary.items():
            print(f"{column}: Z-score: {outliers_dict['z_score_outliers']}, "
                  f"IQR: {outliers_dict['iqr_outliers']}, "
                  f"Modified Z: {outliers_dict['modified_z_outliers']}")
    
    def _profile_consistency(self):
        """Analyze data consistency"""
        consistency_issues = []
        
        # Check for consistent data types within columns
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                non_null_values = self.data[column].dropna()
                if len(non_null_values) > 0:
                    # Check for mixed types
                    types_found = set()
                    for value in non_null_values.head(1000):  # Sample for performance
                        types_found.add(type(value).__name__)
                    
                    if len(types_found) > 1:
                        consistency_issues.append(f"Column '{column}' contains mixed data types: {types_found}")
        
        # Check for consistent formatting in string columns
        string_columns = self.data.select_dtypes(include=['object']).columns
        for column in string_columns:
            non_null_values = self.data[column].dropna()
            if len(non_null_values) > 0:
                # Check for consistent case
                has_upper = any(str(val).isupper() for val in non_null_values.head(100))
                has_lower = any(str(val).islower() for val in non_null_values.head(100))
                has_mixed = any(str(val) != str(val).upper() and str(val) != str(val).lower() 
                               for val in non_null_values.head(100))
                
                if sum([has_upper, has_lower, has_mixed]) > 1:
                    consistency_issues.append(f"Column '{column}' has inconsistent case formatting")
        
        self.profile_results['consistency'] = consistency_issues
        self.quality_issues.extend(consistency_issues)
        
        if consistency_issues:
            print(f"\nCONSISTENCY ISSUES:")
            for issue in consistency_issues:
                print(f"  ⚠️ {issue}")
    
    def _profile_business_rules(self):
        """Validate business rules specific to financial data"""
        business_rule_violations = []
        
        # Check for negative prices (if price columns exist)
        price_columns = [col for col in self.data.columns 
                        if any(keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low'])]
        
        for column in price_columns:
            if self.data[column].dtype in ['float64', 'int64']:
                negative_prices = (self.data[column] < 0).sum()
                if negative_prices > 0:
                    business_rule_violations.append(f"Column '{column}' has {negative_prices} negative values")
        
        # Check for volume consistency (volume should be non-negative)
        volume_columns = [col for col in self.data.columns 
                         if any(keyword in col.lower() for keyword in ['volume', 'shares', 'quantity'])]
        
        for column in volume_columns:
            if self.data[column].dtype in ['float64', 'int64']:
                negative_volumes = (self.data[column] < 0).sum()
                if negative_volumes > 0:
                    business_rule_violations.append(f"Column '{column}' has {negative_volumes} negative values")
        
        # Check for percentage values outside reasonable ranges
        percentage_columns = [col for col in self.data.columns 
                             if any(keyword in col.lower() for keyword in ['return', 'yield', 'rate', 'pct', '%'])]
        
        for column in percentage_columns:
            if self.data[column].dtype in ['float64', 'int64']:
                extreme_values = ((self.data[column] < -1) | (self.data[column] > 10)).sum()
                if extreme_values > 0:
                    business_rule_violations.append(
                        f"Column '{column}' has {extreme_values} values outside reasonable range (-100% to 1000%)"
                    )
        
        self.profile_results['business_rules'] = business_rule_violations
        self.quality_issues.extend(business_rule_violations)
        
        if business_rule_violations:
            print(f"\nBUSINESS RULE VIOLATIONS:")
            for violation in business_rule_violations:
                print(f"  ⚠️ {violation}")
    
    def _generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n" + "="*50)
        print(f"DATA QUALITY SUMMARY REPORT")
        print(f"="*50)
        
        # Overall quality score
        total_issues = len(self.quality_issues)
        total_columns = len(self.data.columns)
        
        # Calculate weighted quality score
        completeness_score = self.profile_results['completeness']['Completeness_Rate']
        
        # Outlier penalty
        outlier_penalty = 0
        if 'outliers' in self.profile_results:
            total_outliers = sum(
                outliers['iqr_outliers'] for outliers in self.profile_results['outliers'].values()
            )
            outlier_penalty = min(20, (total_outliers / len(self.data)) * 100)
        
        # Issue penalty
        issue_penalty = min(30, (total_issues / total_columns) * 100)
        
        overall_quality_score = max(0, completeness_score - outlier_penalty - issue_penalty)
        
        print(f"\nOVERALL QUALITY SCORE: {overall_quality_score:.1f}/100")
        
        if overall_quality_score >= 90:
            quality_rating = "EXCELLENT"
        elif overall_quality_score >= 80:
            quality_rating = "GOOD"
        elif overall_quality_score >= 70:
            quality_rating = "FAIR"
        elif overall_quality_score >= 60:
            quality_rating = "POOR"
        else:
            quality_rating = "CRITICAL"
        
        print(f"QUALITY RATING: {quality_rating}")
        
        print(f"\nQUALITY BREAKDOWN:")
        print(f"  Completeness Score: {completeness_score:.1f}%")
        print(f"  Outlier Penalty: -{outlier_penalty:.1f}")
        print(f"  Issue Penalty: -{issue_penalty:.1f}")
        
        print(f"\nTOTAL QUALITY ISSUES IDENTIFIED: {total_issues}")
        
        if self.quality_issues:
            print(f"\nTOP PRIORITY ISSUES:")
            for i, issue in enumerate(self.quality_issues[:10], 1):
                print(f"  {i}. {issue}")
            
            if len(self.quality_issues) > 10:
                print(f"  ... and {len(self.quality_issues) - 10} more issues")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if completeness_score < 95:
            print(f"  • Address missing data issues (current completeness: {completeness_score:.1f}%)")
        if outlier_penalty > 5:
            print(f"  • Investigate and clean outliers (penalty: {outlier_penalty:.1f})")
        if total_issues > 5:
            print(f"  • Implement systematic data validation procedures")
        if any('negative' in issue for issue in self.quality_issues):
            print(f"  • Review business rule validation for financial constraints")
        
        self.profile_results['summary'] = {
            'overall_quality_score': overall_quality_score,
            'quality_rating': quality_rating,
            'total_issues': total_issues,
            'completeness_score': completeness_score,
            'outlier_penalty': outlier_penalty,
            'issue_penalty': issue_penalty
        }

def create_realistic_financial_dataset():
    """Create a realistic financial dataset with various quality issues"""
    np.random.seed(42)
    
    # Generate base data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Stock price simulation with realistic issues
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices[1:])  # Remove initial price
    
    # Create DataFrame with various quality issues
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.001, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0.002, 0.003, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0.002, 0.003, n_days))),
        'Close': prices,
        'Volume': np.random.lognormal(15, 0.5, n_days).astype(int),
        'Adj_Close': prices * (1 + np.random.normal(0, 0.0001, n_days))
    })
    
    # Introduce quality issues
    # 1. Missing data
    missing_indices = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)
    data.loc[missing_indices, 'Volume'] = np.nan
    
    # 2. Outliers
    outlier_indices = np.random.choice(n_days, size=int(n_days * 0.001), replace=False)
    data.loc[outlier_indices, 'Volume'] *= 50  # Extreme volume spikes
    
    # 3. Negative values (data error)
    error_indices = np.random.choice(n_days, size=3, replace=False)
    data.loc[error_indices, 'Low'] *= -1  # Negative prices (clearly an error)
    
    # 4. Inconsistent high/low relationships
    inconsistent_indices = np.random.choice(n_days, size=5, replace=False)
    data.loc[inconsistent_indices, 'High'] = data.loc[inconsistent_indices, 'Low'] * 0.9
    
    # 5. Duplicate dates
    duplicate_date = dates[100]
    duplicate_row = data.iloc[100].copy()
    data = pd.concat([data, pd.DataFrame([duplicate_row])], ignore_index=True)
    
    # 6. Format inconsistencies (mixed case in a hypothetical symbol column)
    symbols = ['AAPL'] * (n_days // 2) + ['aapl'] * (n_days - n_days // 2) + ['AAPL']  # Mixed case
    data['Symbol'] = symbols
    
    return data.set_index('Date')

def main():
    """Main function to demonstrate the financial data profiler"""
    
    print("Creating realistic financial dataset with quality issues...")
    financial_data = create_realistic_financial_dataset()
    print(f"Dataset created with shape: {financial_data.shape}")
    print(f"Date range: {financial_data.index.min()} to {financial_data.index.max()}")
    
    # Initialize and run profiler
    profiler = FinancialDataProfiler(financial_data)
    profile_results = profiler.generate_comprehensive_profile()
    
    return profile_results

if __name__ == "__main__":
    main()

