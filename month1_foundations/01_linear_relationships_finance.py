"""
Linear Relationships in Finance - Code from Financial Engineering Textbook
=========================================================================

This module demonstrates linear relationships in financial analysis including:
- Portfolio value growth with regular contributions
- Capital Asset Pricing Model (CAPM) relationships
- Bond duration approximations
- Cost structure analysis

Author: Extracted from Financial Engineering with AI and Blockchain textbook
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Demonstrate linear relationships in finance
np.random.seed(42)

# Example 1: Portfolio Value Growth with Regular Contributions
months = np.arange(0, 61)  # 5 years of monthly data
initial_investment = 10000
monthly_contribution = 500
monthly_return = 0.007  # 0.7% monthly return (about 8.7% annually)

# Linear growth (no compounding for simplicity)
linear_portfolio_value = initial_investment + (monthly_contribution * months)

# More realistic: compound growth
compound_portfolio_value = [initial_investment]
for month in range(1, len(months)):
    previous_value = compound_portfolio_value[-1]
    new_contribution = monthly_contribution
    growth = (previous_value + new_contribution) * monthly_return
    compound_portfolio_value.append(previous_value + new_contribution + growth)

plt.figure(figsize=(14, 10))

# Plot 1: Linear vs Compound Growth
plt.subplot(2, 2, 1)
plt.plot(months, linear_portfolio_value, label='Linear Growth', linewidth=2, color='blue')
plt.plot(months, compound_portfolio_value, label='Compound Growth', linewidth=2, color='green')
plt.xlabel('Months')
plt.ylabel('Portfolio Value ($)')
plt.title('Linear vs Compound Portfolio Growth')
plt.legend()
plt.grid(True, alpha=0.3)

# Example 2: CAPM Relationship
market_returns = np.random.normal(0.08, 0.15, 100)  # Market returns: 8% mean, 15% volatility
risk_free_rate = 0.03
beta_values = [0.5, 1.0, 1.5, 2.0]

plt.subplot(2, 2, 2)
for beta in beta_values:
    expected_returns = risk_free_rate + beta * (market_returns - risk_free_rate)
    plt.scatter(market_returns, expected_returns, alpha=0.6, label=f'β = {beta}')

plt.xlabel('Market Return')
plt.ylabel('Expected Stock Return')
plt.title('Capital Asset Pricing Model (CAPM)')
plt.legend()
plt.grid(True, alpha=0.3)

# Example 3: Bond Duration (Linear Approximation)
interest_rates = np.linspace(0.01, 0.10, 100)
bond_duration = 5  # 5-year duration
initial_rate = 0.05
initial_price = 100

# Linear approximation: ΔP/P ≈ -Duration × Δr
price_changes = -bond_duration * (interest_rates - initial_rate)
bond_prices = initial_price * (1 + price_changes)

plt.subplot(2, 2, 3)
plt.plot(interest_rates * 100, bond_prices, linewidth=2, color='red')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Bond Price')
plt.title('Bond Price vs Interest Rate (Duration Approximation)')
plt.grid(True, alpha=0.3)

# Example 4: Cost Structure Analysis
production_units = np.arange(0, 1001, 50)
fixed_costs = 50000
variable_cost_per_unit = 25
total_costs = fixed_costs + variable_cost_per_unit * production_units

plt.subplot(2, 2, 4)
plt.plot(production_units, total_costs, linewidth=2, color='purple')
plt.xlabel('Production Units')
plt.ylabel('Total Cost ($)')
plt.title('Linear Cost Structure')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print insights
print("LINEAR RELATIONSHIPS IN FINANCE - KEY INSIGHTS:")
print("=" * 50)
print(f"1. Portfolio Growth:")
print(f"   Linear (5 years): ${linear_portfolio_value[-1]:,.2f}")
print(f"   Compound (5 years): ${compound_portfolio_value[-1]:,.2f}")
print(f"   Difference: ${compound_portfolio_value[-1] - linear_portfolio_value[-1]:,.2f}")
print()
print(f"2. Cost Structure Analysis:")
print(f"   Fixed Costs: ${fixed_costs:,}")
print(f"   Variable Cost per Unit: ${variable_cost_per_unit}")
print(f"   Break-even point (assuming $50/unit revenue): {fixed_costs / (50 - variable_cost_per_unit):.0f} units")

if __name__ == "__main__":
    print("Running Linear Relationships in Finance Analysis...")

