# Financial Engineering with AI and Blockchain - Python Code Repository by Dr Nikhil Varma

This repository contains Python code extracted from the comprehensive textbook "Financial Engineering with AI and Blockchain: A 24-Month Learning Journey for High School Students". The code has been organized into modules that demonstrate key concepts in financial analysis, data science, and quantitative finance.

## Repository Structure

```
FEBook/
├── README.md
└── month1_foundations/
    ├── 01_linear_relationships_finance.py
    ├── 02_descriptive_statistics_analysis.py
    ├── 03_probability_risk_analysis.py
    ├── 04_numpy_financial_analysis.py
    ├── 05_financial_data_profiler.py
    └── 06_financial_data_cleaner.py
```

## Month 1: Building Your Foundation

The `month1_foundations/` directory contains Python modules covering the fundamental concepts needed for financial analysis and data science.

### Module Descriptions

#### 1. Linear Relationships in Finance (`01_linear_relationships_finance.py`)
- **Purpose**: Demonstrates linear relationships in financial analysis
- **Key Features**:
  - Portfolio value growth with regular contributions
  - Capital Asset Pricing Model (CAPM) relationships
  - Bond duration approximations using linear models
  - Cost structure analysis for businesses
- **Dependencies**: `numpy`, `matplotlib`, `pandas`

#### 2. Descriptive Statistics Analysis (`02_descriptive_statistics_analysis.py`)
- **Purpose**: Comprehensive statistical analysis of financial data
- **Key Features**:
  - Analysis of different asset classes (Large-cap, Small-cap, Emerging markets, Merger arbitrage)
  - Central tendency measures (mean, median, mode)
  - Variability measures (standard deviation, variance, skewness, kurtosis)
  - Risk-return analysis and visualization
  - Distribution characteristics and normality testing
- **Dependencies**: `numpy`, `matplotlib`, `pandas`, `scipy`, `seaborn`

#### 3. Probability and Risk Analysis (`03_probability_risk_analysis.py`)
- **Purpose**: Advanced probability theory and risk measurement in finance
- **Key Features**:
  - Market scenario modeling (Bull, Normal, Bear, Crisis markets)
  - Monte Carlo simulations for portfolio analysis
  - Comprehensive risk metrics (VaR, CVaR, Maximum Drawdown)
  - Scenario-based return analysis
  - Portfolio evolution visualization
- **Dependencies**: `numpy`, `matplotlib`, `pandas`, `scipy`, `seaborn`

#### 4. NumPy for Financial Analysis (`04_numpy_financial_analysis.py`)
- **Purpose**: Demonstrates NumPy's mathematical capabilities for finance
- **Key Features**:
  - Vectorized financial calculations
  - Performance comparison (vectorized vs. loop-based operations)
  - Advanced NumPy features (broadcasting, linear algebra)
  - Monte Carlo portfolio simulations
  - Risk metrics calculation using NumPy
- **Dependencies**: `numpy`, `time`

#### 5. Financial Data Profiler (`05_financial_data_profiler.py`)
- **Purpose**: Comprehensive data quality analysis for financial datasets
- **Key Features**:
  - Data completeness and missing data analysis
  - Statistical distribution profiling
  - Outlier detection using multiple methods
  - Business rule validation for financial data
  - Data consistency checks
  - Quality scoring and recommendations
- **Dependencies**: `numpy`, `pandas`, `scipy`, `datetime`, `typing`

#### 6. Financial Data Cleaner (`06_financial_data_cleaner.py`)
- **Purpose**: Systematic data cleaning for financial datasets
- **Key Features**:
  - Duplicate record removal
  - Missing data imputation strategies
  - Outlier cleaning using IQR and Z-score methods
  - Business rule enforcement
  - Format standardization
  - Complete audit trail of cleaning operations
- **Dependencies**: `numpy`, `pandas`, `scipy`, `datetime`, `typing`

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Packages
Install the required packages using pip:

```bash
pip install numpy pandas matplotlib scipy seaborn
```

Or using conda:

```bash
conda install numpy pandas matplotlib scipy seaborn
```

### Running the Code

Each module can be run independently:

```bash
# Navigate to the month1_foundations directory
cd month1_foundations/

# Run individual modules
python 01_linear_relationships_finance.py
python 02_descriptive_statistics_analysis.py
python 03_probability_risk_analysis.py
python 04_numpy_financial_analysis.py
python 05_financial_data_profiler.py
python 06_financial_data_cleaner.py
```

## Key Learning Objectives

By working through these modules, you will learn:

1. **Mathematical Foundations**: Understanding linear relationships, exponential growth, and statistical measures in financial contexts

2. **Data Analysis Skills**: Profiling data quality, identifying issues, and implementing systematic cleaning procedures

3. **Risk Management**: Calculating and interpreting various risk metrics including VaR, CVaR, and drawdown analysis

4. **Probability Theory**: Applying Monte Carlo simulations and scenario analysis to financial problems

5. **Performance Optimization**: Using vectorized operations and efficient NumPy techniques for financial calculations

6. **Professional Practices**: Implementing audit trails, validation procedures, and systematic approaches to financial data analysis

## Educational Context

This code is designed for high school students beginning their journey into financial engineering, but it's also valuable for:

- Undergraduate students in finance, economics, or data science
- Professionals transitioning into quantitative finance
- Anyone interested in applying data science techniques to financial markets

## Code Quality and Best Practices

The code follows professional standards including:

- **Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Robust error checking and validation
- **Modularity**: Well-organized, reusable functions and classes
- **Performance**: Efficient algorithms and vectorized operations
- **Reproducibility**: Consistent random seeds and clear methodology

## Future Additions

This repository will be expanded with additional months of content, including:

- Month 2: Advanced model evaluation and robustness
- Month 3: Demand elasticity analysis
- Month 4: Blockchain and DeFi fundamentals
- And continuing through the full 24-month curriculum

## Contributing

This code is extracted from educational material. If you find issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This code is provided for educational purposes. Please refer to the original textbook for complete context and theoretical background.

## Acknowledgments

Code from "Financial Engineering with AI and Blockchain: A 24-Month Learning Journey for High School Students" - a comprehensive educational resource bridging traditional finance with modern technology.

---

**Note**: This repository contains practical implementations of financial concepts. Always validate results and understand the assumptions behind each model before applying to real financial decisions.

