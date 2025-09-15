"""
Comprehensive Probability and Risk Analysis for Financial Markets
================================================================

This module demonstrates probability theory and risk measurement in finance,
including scenario-based analysis, Monte Carlo simulations, and comprehensive
risk metrics calculation.

Author: Extracted from Financial Engineering with AI and Blockchain textbook
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# Advanced probability and risk analysis
np.random.seed(42)

def create_market_scenarios():
    """Create realistic market scenarios with different probability distributions"""
    scenarios = {
        'Bull Market': {
            'probability': 0.25,
            'return_dist': lambda n: np.random.normal(0.15, 0.12, n),  # 15% return, 12% vol
            'description': 'Strong economic growth, low interest rates'
        },
        'Normal Market': {
            'probability': 0.50,
            'return_dist': lambda n: np.random.normal(0.08, 0.16, n),  # 8% return, 16% vol
            'description': 'Moderate growth, stable conditions'
        },
        'Bear Market': {
            'probability': 0.20,
            'return_dist': lambda n: np.random.normal(-0.10, 0.25, n),  # -10% return, 25% vol
            'description': 'Economic contraction, high uncertainty'
        },
        'Crisis': {
            'probability': 0.05,
            'return_dist': lambda n: np.random.normal(-0.35, 0.40, n),  # -35% return, 40% vol
            'description': 'Financial crisis, extreme volatility'
        }
    }
    return scenarios

def simulate_scenario_returns(scenarios, n_simulations, n_years):
    """Simulate returns under different market scenarios"""
    all_returns = []
    scenario_labels = []
    
    for _ in range(n_simulations):
        annual_returns = []
        scenario_sequence = []
        
        for year in range(n_years):
            # Choose scenario based on probabilities
            scenario_choice = np.random.choice(
                list(scenarios.keys()),
                p=[scenarios[s]['probability'] for s in scenarios.keys()]
            )
            
            # Generate return for this scenario
            annual_return = scenarios[scenario_choice]['return_dist'](1)[0]
            annual_returns.append(annual_return)
            scenario_sequence.append(scenario_choice)
        
        all_returns.append(annual_returns)
        scenario_labels.append(scenario_sequence)
    
    return np.array(all_returns), scenario_labels

def calculate_risk_measures(returns_matrix, portfolio_values, confidence_levels=[0.95, 0.99]):
    """Calculate comprehensive risk measures"""
    # Final portfolio values
    final_values = portfolio_values[:, -1]
    
    # Calculate returns
    total_returns = (final_values / portfolio_values[:, 0]) - 1
    annualized_returns = (final_values / portfolio_values[:, 0]) ** (1/n_years) - 1
    
    risk_measures = {}
    
    # Basic statistics
    risk_measures['expected_final_value'] = np.mean(final_values)
    risk_measures['expected_total_return'] = np.mean(total_returns)
    risk_measures['expected_annualized_return'] = np.mean(annualized_returns)
    risk_measures['volatility'] = np.std(annualized_returns)
    
    # Value at Risk (VaR)
    for conf in confidence_levels:
        var_level = (1 - conf) * 100
        var_return = np.percentile(total_returns, var_level)
        var_value = np.percentile(final_values, var_level)
        
        risk_measures[f'VaR_{int(conf*100)}%_return'] = var_return
        risk_measures[f'VaR_{int(conf*100)}%_value'] = var_value
        
        # Conditional VaR (Expected Shortfall)
        cvar_return = np.mean(total_returns[total_returns <= var_return])
        cvar_value = np.mean(final_values[final_values <= var_value])
        
        risk_measures[f'CVaR_{int(conf*100)}%_return'] = cvar_return
        risk_measures[f'CVaR_{int(conf*100)}%_value'] = cvar_value
    
    # Maximum Drawdown
    running_max = np.maximum.accumulate(portfolio_values, axis=1)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdowns = np.min(drawdowns, axis=1)
    risk_measures['max_drawdown'] = np.mean(max_drawdowns)
    risk_measures['worst_drawdown'] = np.min(max_drawdowns)
    
    # Probability of loss
    risk_measures['prob_loss'] = np.mean(total_returns < 0)
    risk_measures['prob_large_loss'] = np.mean(total_returns < -0.20)  # >20% loss
    
    return risk_measures

# Generate scenario-based returns
scenarios = create_market_scenarios()
n_simulations = 10000
n_years = 20

print("Market Scenarios:")
print("=" * 40)
for name, scenario in scenarios.items():
    print(f"{name}: {scenario['probability']:.0%} probability")
    print(f"  {scenario['description']}")
print()

# Run simulation
simulated_returns, scenario_sequences = simulate_scenario_returns(scenarios, n_simulations, n_years)

# Calculate portfolio values over time
initial_investment = 100000
portfolio_values = np.zeros((n_simulations, n_years + 1))
portfolio_values[:, 0] = initial_investment

for sim in range(n_simulations):
    for year in range(n_years):
        portfolio_values[sim, year + 1] = portfolio_values[sim, year] * (1 + simulated_returns[sim, year])

# Calculate comprehensive risk measures
risk_metrics = calculate_risk_measures(simulated_returns, portfolio_values)

# Display results
print("COMPREHENSIVE RISK ANALYSIS RESULTS:")
print("=" * 50)
print(f"Initial Investment: ${initial_investment:,}")
print(f"Time Horizon: {n_years} years")
print(f"Number of Simulations: {n_simulations:,}")
print()

print("EXPECTED OUTCOMES:")
print("-" * 20)
print(f"Expected Final Value: ${risk_metrics['expected_final_value']:,.0f}")
print(f"Expected Total Return: {risk_metrics['expected_total_return']:.1%}")
print(f"Expected Annualized Return: {risk_metrics['expected_annualized_return']:.1%}")
print(f"Annualized Volatility: {risk_metrics['volatility']:.1%}")
print()

print("VALUE AT RISK (VaR) MEASURES:")
print("-" * 30)
for conf in [95, 99]:
    print(f"{conf}% VaR:")
    print(f"  Return: {risk_metrics[f'VaR_{conf}%_return']:.1%}")
    print(f"  Portfolio Value: ${risk_metrics[f'VaR_{conf}%_value']:,.0f}")
    print(f"  Expected Shortfall (CVaR): {risk_metrics[f'CVaR_{conf}%_return']:.1%}")
print()

print("DRAWDOWN ANALYSIS:")
print("-" * 18)
print(f"Average Maximum Drawdown: {risk_metrics['max_drawdown']:.1%}")
print(f"Worst Case Drawdown: {risk_metrics['worst_drawdown']:.1%}")
print()

print("LOSS PROBABILITIES:")
print("-" * 18)
print(f"Probability of Any Loss: {risk_metrics['prob_loss']:.1%}")
print(f"Probability of >20% Loss: {risk_metrics['prob_large_loss']:.1%}")

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Distribution of final values
axes[0, 0].hist(portfolio_values[:, -1], bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(risk_metrics['expected_final_value'], color='red', linestyle='--', 
                   label=f"Expected: ${risk_metrics['expected_final_value']:,.0f}")
axes[0, 0].axvline(risk_metrics['VaR_95%_value'], color='orange', linestyle='--',
                   label=f"95% VaR: ${risk_metrics['VaR_95%_value']:,.0f}")
axes[0, 0].set_xlabel('Final Portfolio Value ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Final Portfolio Values')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Sample portfolio paths
sample_paths = portfolio_values[:100, :]  # Show first 100 paths
years = np.arange(n_years + 1)
for i in range(100):
    axes[0, 1].plot(years, sample_paths[i, :], alpha=0.1, color='blue')

# Add percentile paths
percentiles = [5, 25, 50, 75, 95]
colors = ['red', 'orange', 'green', 'orange', 'red']
for p, color in zip(percentiles, colors):
    path = np.percentile(portfolio_values, p, axis=0)
    axes[0, 1].plot(years, path, color=color, linewidth=2, label=f'{p}th percentile')

axes[0, 1].set_xlabel('Years')
axes[0, 1].set_ylabel('Portfolio Value ($)')
axes[0, 1].set_title('Portfolio Value Evolution (Sample Paths)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Annual returns distribution
annual_returns = simulated_returns.flatten()
axes[0, 2].hist(annual_returns, bins=50, alpha=0.7, edgecolor='black')
axes[0, 2].axvline(np.mean(annual_returns), color='red', linestyle='--', 
                   label=f"Mean: {np.mean(annual_returns):.1%}")
axes[0, 2].axvline(np.percentile(annual_returns, 5), color='orange', linestyle='--',
                   label=f"5th percentile: {np.percentile(annual_returns, 5):.1%}")
axes[0, 2].set_xlabel('Annual Return')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Distribution of Annual Returns')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Drawdown analysis
running_max = np.maximum.accumulate(portfolio_values, axis=1)
drawdowns = (portfolio_values - running_max) / running_max
worst_drawdown_path = drawdowns[np.argmin(np.min(drawdowns, axis=1)), :]

axes[1, 0].fill_between(years, 0, worst_drawdown_path, alpha=0.3, color='red', label='Worst Case')
axes[1, 0].plot(years, np.mean(drawdowns, axis=0), color='blue', linewidth=2, label='Average')
axes[1, 0].set_xlabel('Years')
axes[1, 0].set_ylabel('Drawdown (%)')
axes[1, 0].set_title('Portfolio Drawdown Analysis')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Scenario frequency analysis
scenario_counts = {}
for scenario_seq in scenario_sequences:
    for scenario in scenario_seq:
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

total_scenarios = sum(scenario_counts.values())
scenario_frequencies = {k: v/total_scenarios for k, v in scenario_counts.items()}

axes[1, 1].bar(scenario_frequencies.keys(), scenario_frequencies.values(), alpha=0.7)
axes[1, 1].set_xlabel('Market Scenario')
axes[1, 1].set_ylabel('Observed Frequency')
axes[1, 1].set_title('Observed vs Expected Scenario Frequencies')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# Add expected frequencies
expected_freqs = [scenarios[s]['probability'] for s in scenario_frequencies.keys()]
axes[1, 1].plot(list(scenario_frequencies.keys()), expected_freqs, 'ro-', label='Expected')
axes[1, 1].legend()

# 6. Risk-Return scatter
final_returns = (portfolio_values[:, -1] / portfolio_values[:, 0]) ** (1/n_years) - 1
final_volatilities = []

for sim in range(min(1000, n_simulations)):  # Sample for performance
    vol = np.std(simulated_returns[sim, :])
    final_volatilities.append(vol)

axes[1, 2].scatter(final_volatilities, final_returns[:len(final_volatilities)], alpha=0.5)
axes[1, 2].set_xlabel('Realized Volatility')
axes[1, 2].set_ylabel('Annualized Return')
axes[1, 2].set_title('Risk-Return Relationship')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary insights
print("\n" + "="*60)
print("KEY INSIGHTS FROM PROBABILITY AND RISK ANALYSIS:")
print("="*60)
print("\n1. EXPECTED OUTCOMES:")
print(f"   • Over {n_years} years, expect portfolio to grow to ~${risk_metrics['expected_final_value']:,.0f}")
print(f"   • This represents a {risk_metrics['expected_annualized_return']:.1%} annualized return")

print("\n2. DOWNSIDE RISK:")
print(f"   • 5% chance of losing more than {abs(risk_metrics['VaR_95%_return']):.1%}")
print(f"   • In worst 5% of cases, expect to lose {abs(risk_metrics['CVaR_95%_return']):.1%} on average")
print(f"   • {risk_metrics['prob_loss']:.1%} probability of ending with a loss")

print("\n3. VOLATILITY AND DRAWDOWNS:")
print(f"   • Expect average maximum drawdown of {abs(risk_metrics['max_drawdown']):.1%}")
print(f"   • Worst case scenario shows drawdown of {abs(risk_metrics['worst_drawdown']):.1%}")

print("\n4. SCENARIO IMPACT:")
print("   • Market regime significantly affects outcomes")
print("   • Crisis scenarios, though rare (5% probability), have major impact")
print("   • Diversification across time helps reduce scenario risk")

if __name__ == "__main__":
    print("Running Comprehensive Probability and Risk Analysis...")

