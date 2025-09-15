"""
Comprehensive Descriptive Statistics Analysis for Financial Data
===============================================================

This module provides comprehensive analysis of descriptive statistics in finance,
including measures of central tendency, variability, and distribution characteristics
for different types of financial assets.

Author: Extracted from Financial Engineering with AI and Blockchain textbook
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Generate realistic financial data with different characteristics
np.random.seed(42)
n_observations = 1000

# Dataset 1: Normal-like returns (large-cap stock)
normal_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), n_observations)

# Dataset 2: Skewed returns (small-cap stock with occasional large gains)
skewed_base = np.random.normal(0.10/252, 0.25/np.sqrt(252), n_observations)
# Add occasional large positive returns (momentum/growth spurts)
large_gains = np.random.choice([0, 1], size=n_observations, p=[0.95, 0.05])
skewed_returns = skewed_base + large_gains * np.random.exponential(0.05, n_observations)

# Dataset 3: Fat-tailed returns (emerging market or crypto-like)
# Use t-distribution for fat tails
fat_tailed_returns = stats.t.rvs(df=3, loc=0.12/252, scale=0.30/np.sqrt(252), size=n_observations)

# Dataset 4: Bimodal returns (merger arbitrage or binary outcome strategy)
mode1 = np.random.normal(-0.02, 0.01, n_observations//2)
mode2 = np.random.normal(0.08, 0.01, n_observations//2)
bimodal_returns = np.concatenate([mode1, mode2])
np.random.shuffle(bimodal_returns)

# Create a comprehensive analysis
datasets = {
    'Large-Cap Stock': normal_returns,
    'Small-Cap Growth': skewed_returns,
    'Emerging Market': fat_tailed_returns,
    'Merger Arbitrage': bimodal_returns
}

def calculate_comprehensive_stats(data):
    """Calculate comprehensive descriptive statistics"""
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': stats.mode(data, keepdims=True)[0][0] if len(stats.mode(data, keepdims=True)[0]) > 0 else np.nan,
        'std': np.std(data, ddof=1),
        'variance': np.var(data, ddof=1),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'var_95': np.percentile(data, 5),
        'var_99': np.percentile(data, 1)
    }

# Calculate statistics for all datasets
stats_summary = {}
for name, data in datasets.items():
    stats_summary[name] = calculate_comprehensive_stats(data)

# Create comprehensive visualization
fig, axes = plt.subplots(4, 4, figsize=(20, 16))

for i, (name, data) in enumerate(datasets.items()):
    # Histogram with statistics
    axes[i, 0].hist(data, bins=50, alpha=0.7, density=True, edgecolor='black')
    
    # Add vertical lines for mean, median
    mean_val = stats_summary[name]['mean']
    median_val = stats_summary[name]['median']
    axes[i, 0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    axes[i, 0].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    axes[i, 0].set_title(f'{name} - Distribution')
    axes[i, 0].set_xlabel('Daily Return')
    axes[i, 0].set_ylabel('Density')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[i, 1].boxplot(data, vert=True)
    axes[i, 1].set_title(f'{name} - Box Plot')
    axes[i, 1].set_ylabel('Daily Return')
    axes[i, 1].grid(True, alpha=0.3)
    
    # Q-Q plot against normal distribution
    stats.probplot(data, dist="norm", plot=axes[i, 2])
    axes[i, 2].set_title(f'{name} - Q-Q Plot (Normal)')
    axes[i, 2].grid(True, alpha=0.3)
    
    # Time series plot
    axes[i, 3].plot(data, linewidth=0.5, alpha=0.7)
    axes[i, 3].set_title(f'{name} - Time Series')
    axes[i, 3].set_xlabel('Observation')
    axes[i, 3].set_ylabel('Daily Return')
    axes[i, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create a comprehensive statistics table
stats_df = pd.DataFrame(stats_summary).T

# Display the statistics table
print("COMPREHENSIVE DESCRIPTIVE STATISTICS ANALYSIS:")
print("=" * 80)
print("\nDETAILED STATISTICS TABLE:")
print("-" * 50)

# Format the statistics nicely
formatted_stats = stats_df.copy()
for col in ['mean', 'median', 'std', 'variance', 'min', 'max', 'range', 'q25', 'q75', 'iqr', 'var_95', 'var_99']:
    if col in formatted_stats.columns:
        formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.6f}")

for col in ['skewness', 'kurtosis']:
    if col in formatted_stats.columns:
        formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.3f}")

print(formatted_stats.to_string())

# Risk-Return Analysis
print(f"\n\nRISK-RETURN ANALYSIS:")
print("-" * 25)
risk_return_data = []

for name, data in datasets.items():
    annual_return = np.mean(data) * 252
    annual_volatility = np.std(data, ddof=1) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    risk_return_data.append({
        'Asset': name,
        'Annual Return': f"{annual_return:.2%}",
        'Annual Volatility': f"{annual_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.3f}"
    })

risk_return_df = pd.DataFrame(risk_return_data)
print(risk_return_df.to_string(index=False))

# Distribution Analysis
print(f"\n\nDISTRIBUTION CHARACTERISTICS:")
print("-" * 35)

for name, data in datasets.items():
    stats_dict = stats_summary[name]
    print(f"\n{name}:")
    print(f"  Central Tendency:")
    print(f"    Mean vs Median difference: {abs(stats_dict['mean'] - stats_dict['median']):.6f}")
    
    if stats_dict['skewness'] > 0.5:
        skew_desc = "Positively skewed (right tail)"
    elif stats_dict['skewness'] < -0.5:
        skew_desc = "Negatively skewed (left tail)"
    else:
        skew_desc = "Approximately symmetric"
    
    if stats_dict['kurtosis'] > 3:
        kurt_desc = "Leptokurtic (fat tails)"
    elif stats_dict['kurtosis'] < 3:
        kurt_desc = "Platykurtic (thin tails)"
    else:
        kurt_desc = "Mesokurtic (normal tails)"
    
    print(f"  Shape:")
    print(f"    Skewness: {stats_dict['skewness']:.3f} ({skew_desc})")
    print(f"    Kurtosis: {stats_dict['kurtosis']:.3f} ({kurt_desc})")
    print(f"  Risk Measures:")
    print(f"    95% VaR: {stats_dict['var_95']:.6f}")
    print(f"    99% VaR: {stats_dict['var_99']:.6f}")

# Practical implications
print(f"\n\nPRACTICAL IMPLICATIONS FOR INVESTORS:")
print("-" * 45)
print(f"\n1. CENTRAL TENDENCY MEASURES:")
print(f"   • Use MEAN for: Expected returns, portfolio optimization")
print(f"   • Use MEDIAN for: Typical performance, skewed distributions")
print(f"   • Large mean-median differences indicate skewed returns")

print(f"\n2. VARIABILITY MEASURES:")
print(f"   • Standard deviation = Volatility = Risk measure")
print(f"   • Higher volatility requires higher expected returns")
print(f"   • Volatility clustering is common in financial markets")

print(f"\n3. DISTRIBUTION SHAPE:")
print(f"   • Skewness affects option pricing and risk management")
print(f"   • Kurtosis (fat tails) increases extreme event probability")
print(f"   • Normal distribution assumptions often violated in finance")

print(f"\n4. RISK MANAGEMENT:")
print(f"   • VaR measures potential losses at given confidence levels")
print(f"   • Fat-tailed distributions have higher tail risks")
print(f"   • Diversification benefits depend on correlation structure")

# Create a risk-return scatter plot
plt.figure(figsize=(12, 8))
returns_annual = [np.mean(data) * 252 for data in datasets.values()]
volatilities_annual = [np.std(data, ddof=1) * np.sqrt(252) for data in datasets.values()]
names = list(datasets.keys())
colors = ['blue', 'green', 'red', 'purple']

for i, (ret, vol, name) in enumerate(zip(returns_annual, volatilities_annual, names)):
    plt.scatter(vol, ret, s=200, c=colors[i], alpha=0.7, label=name)
    plt.annotate(name, (vol, ret), xytext=(10, 10), textcoords='offset points')

plt.xlabel('Annual Volatility (Risk)')
plt.ylabel('Annual Expected Return')
plt.title('Risk-Return Profile of Different Asset Classes')
plt.legend()
plt.grid(True, alpha=0.3)

# Add efficient frontier concept
vol_range = np.linspace(min(volatilities_annual), max(volatilities_annual), 100)
# Simplified efficient frontier (for illustration)
efficient_returns = 0.02 + 0.3 * vol_range + 0.1 * vol_range**2
plt.plot(vol_range, efficient_returns, 'k--', alpha=0.5, label='Theoretical Efficient Frontier')
plt.legend()
plt.show()

if __name__ == "__main__":
    print("Running Comprehensive Descriptive Statistics Analysis...")

