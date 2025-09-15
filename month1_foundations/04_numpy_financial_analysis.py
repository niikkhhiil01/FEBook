"""
NumPy for Financial Analysis - Mathematical Foundation
=====================================================

This module demonstrates NumPy's mathematical capabilities for financial analysis,
including vectorized operations, performance comparisons, and advanced financial
calculations using NumPy arrays.

Author: Extracted from Financial Engineering with AI and Blockchain textbook
"""

import numpy as np
import time

def demonstrate_numpy_capabilities():
    """Demonstration of NumPy's mathematical capabilities for finance"""
    
    # Financial time series analysis with NumPy
    np.random.seed(42)
    
    # Generate realistic stock price data
    n_days = 252 * 5  # 5 years of daily data
    initial_price = 100.0
    daily_volatility = 0.02  # 2% daily volatility
    annual_drift = 0.08  # 8% annual expected return
    daily_drift = annual_drift / 252
    
    # Geometric Brownian Motion simulation
    random_shocks = np.random.normal(0, daily_volatility, n_days)
    log_returns = daily_drift + random_shocks
    cumulative_log_returns = np.cumsum(log_returns)
    prices = initial_price * np.exp(cumulative_log_returns)
    
    print("NumPy Array Properties:")
    print(f"Data type: {prices.dtype}")
    print(f"Shape: {prices.shape}")
    print(f"Memory usage: {prices.nbytes} bytes")
    print(f"Is contiguous: {prices.flags['C_CONTIGUOUS']}")
    
    # Vectorized financial calculations
    print("\nVectorized Financial Calculations:")
    
    # Simple returns
    simple_returns = np.diff(prices) / prices[:-1]
    
    # Log returns (more mathematically convenient)
    log_returns_calculated = np.diff(np.log(prices))
    
    # Moving averages using convolution
    window_size = 20
    moving_average = np.convolve(prices, np.ones(window_size)/window_size, mode='valid')
    
    # Volatility calculation (rolling standard deviation)
    def rolling_std(data, window):
        """Calculate rolling standard deviation using NumPy"""
        result = np.empty(len(data) - window + 1)
        for i in range(len(result)):
            result[i] = np.std(data[i:i+window], ddof=1)
        return result
    
    rolling_volatility = rolling_std(log_returns_calculated, 30) * np.sqrt(252)  # Annualized
    
    return prices, simple_returns, log_returns_calculated, moving_average, rolling_volatility

def performance_comparison(prices):
    """Performance comparison: vectorized vs loop-based calculations"""
    
    def calculate_returns_loop(prices):
        """Calculate returns using explicit loop (inefficient)"""
        returns = np.empty(len(prices) - 1)
        for i in range(1, len(prices)):
            returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
        return returns
    
    def calculate_returns_vectorized(prices):
        """Calculate returns using vectorized operations (efficient)"""
        return np.diff(prices) / prices[:-1]
    
    # Timing comparison
    start_time = time.time()
    returns_loop = calculate_returns_loop(prices)
    loop_time = time.time() - start_time
    
    start_time = time.time()
    returns_vectorized = calculate_returns_vectorized(prices)
    vectorized_time = time.time() - start_time
    
    print(f"\nPerformance Comparison:")
    print(f"Loop-based calculation: {loop_time:.6f} seconds")
    print(f"Vectorized calculation: {vectorized_time:.6f} seconds")
    print(f"Speedup: {loop_time / vectorized_time:.1f}x")
    
    # Verify results are identical
    print(f"Results identical: {np.allclose(returns_loop, returns_vectorized)}")
    
    return returns_vectorized

def advanced_numpy_features():
    """Advanced NumPy features for finance"""
    
    print("\nAdvanced NumPy Features:")
    
    # Broadcasting for portfolio calculations
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # Portfolio weights
    asset_returns = np.random.normal(0.08/252, 0.02, (252, 4))  # 4 assets, 252 days
    
    # Portfolio returns using broadcasting
    portfolio_returns = np.sum(asset_returns * weights, axis=1)
    print(f"Portfolio return calculation shape: {asset_returns.shape} * {weights.shape} -> {portfolio_returns.shape}")
    
    # Correlation matrix calculation
    correlation_matrix = np.corrcoef(asset_returns.T)
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    
    # Eigenvalue decomposition for principal component analysis
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Explained variance ratio: {eigenvalues / np.sum(eigenvalues)}")
    
    return portfolio_returns, correlation_matrix, eigenvalues, eigenvectors

def calculate_risk_metrics(returns):
    """Risk metrics using NumPy"""
    
    def calculate_var(returns, confidence_level=0.05):
        """Calculate Value at Risk using NumPy percentile function"""
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(returns, confidence_level=0.05):
        """Calculate Conditional Value at Risk"""
        var_threshold = calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var_threshold])
    
    var_95 = calculate_var(returns, 0.05)
    cvar_95 = calculate_cvar(returns, 0.05)
    
    print(f"\nRisk Metrics:")
    print(f"95% VaR: {var_95:.4f}")
    print(f"95% CVaR: {cvar_95:.4f}")
    
    return var_95, cvar_95

def monte_carlo_portfolio_simulation(initial_value, expected_return, volatility, time_horizon, n_simulations):
    """
    Monte Carlo simulation of portfolio value evolution
    
    Parameters:
    -----------
    initial_value : float
        Starting portfolio value
    expected_return : float
        Annual expected return
    volatility : float
        Annual volatility
    time_horizon : float
        Time horizon in years
    n_simulations : int
        Number of simulation paths
    
    Returns:
    --------
    np.ndarray
        Final portfolio values for each simulation
    """
    # Convert to daily parameters
    daily_return = expected_return / 252
    daily_vol = volatility / np.sqrt(252)
    n_days = int(time_horizon * 252)
    
    # Generate random shocks for all simulations at once
    random_shocks = np.random.normal(0, daily_vol, (n_simulations, n_days))
    
    # Calculate cumulative returns
    daily_returns = daily_return + random_shocks
    cumulative_returns = np.cumprod(1 + daily_returns, axis=1)
    
    # Final values
    final_values = initial_value * cumulative_returns[:, -1]
    
    return final_values

def main():
    """Main function to run all demonstrations"""
    
    print("NumPy for Financial Analysis - Comprehensive Demonstration")
    print("=" * 60)
    
    # Demonstrate basic capabilities
    prices, simple_returns, log_returns, moving_average, rolling_volatility = demonstrate_numpy_capabilities()
    
    # Performance comparison
    returns = performance_comparison(prices)
    
    # Advanced features
    portfolio_returns, correlation_matrix, eigenvalues, eigenvectors = advanced_numpy_features()
    
    # Risk metrics
    var_95, cvar_95 = calculate_risk_metrics(portfolio_returns)
    
    # Monte Carlo simulation
    print("\nMonte Carlo Simulation:")
    n_simulations = 10000
    final_values = monte_carlo_portfolio_simulation(
        initial_value=100000,
        expected_return=0.08,
        volatility=0.15,
        time_horizon=5,
        n_simulations=n_simulations
    )
    
    print(f"Monte Carlo Simulation Results ({n_simulations:,} simulations):")
    print(f"Expected final value: ${np.mean(final_values):,.0f}")
    print(f"Standard deviation: ${np.std(final_values):,.0f}")
    print(f"95% confidence interval: ${np.percentile(final_values, 2.5):,.0f} - ${np.percentile(final_values, 97.5):,.0f}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. NumPy provides the mathematical foundation for financial analysis")
    print("2. Vectorized operations are significantly faster than loops")
    print("3. Broadcasting enables efficient portfolio calculations")
    print("4. Linear algebra operations support advanced analytics")
    print("5. Monte Carlo simulations scale efficiently with NumPy")

if __name__ == "__main__":
    main()

