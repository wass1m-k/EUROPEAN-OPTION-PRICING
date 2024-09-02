# EUROPEAN-OPTION-PRICING
This Python module calculates European option prices using Black-Scholes, Merton jump-diffusion, and Binomial Tree models. It also computes option Greeks and implied volatility using numerical methods. Includes tools for visualizing implied volatility surfaces.


---

# European Option Pricing and Implied Volatility

This repository contains a Python implementation for pricing European options and calculating implied volatility using various models. The `EuropeanOption` class supports Black-Scholes, Merton Jump Diffusion, and Binomial Tree models. Additionally, it includes functions for computing implied volatility using least-squares and Newton-Raphson methods. Visualization of implied volatility surfaces is also provided.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [EuropeanOption Class](#europeanoption-class)
  - [Implied Volatility Functions](#implied-volatility-functions)
  - [Plotting Volatility Surfaces](#plotting-volatility-surfaces)
- [Examples](#examples)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Installation

Ensure you have the required libraries installed. You can install them using pip:

```bash
pip install numpy scipy plotly
```

## Usage

### EuropeanOption Class

The `EuropeanOption` class allows you to price European options and compute Greeks. It supports the following models:

- **Black-Scholes Model**
- **Merton Jump Diffusion Model**
- **Binomial Tree Model**

#### Initialization

```python
from option_pricing import EuropeanOption

# Initialize a European call option
option = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, Option_type='call')
```

#### Methods

- `black_and_scholes()`: Calculates the option price using the Black-Scholes model.
- `merton_jump_diffusion(mu_j, sigma_j, lam, max_iter=100, stop_cond=1e-15)`: Prices the option using the Merton Jump Diffusion model.
- `binomial_tree(n)`: Prices the option using the Binomial Tree model.
- `delta()`: Computes the Delta of the option.
- `gamma()`: Computes the Gamma of the option.
- `vega()`: Computes the Vega of the option.
- `theta()`: Computes the Theta of the option.
- `rho()`: Computes the Rho of the option.

#### Example

```python
# Price using Black-Scholes model
price = option.black_and_scholes()
print(f"Option Price: ${price:.2f}")
```

### Implied Volatility Functions

These functions compute the implied volatility from market prices using different methods.

#### `implied_volatility(market_price, S, K, T, r, type)`

Calculates the implied volatility using numerical optimization.

#### `least_squares_iv(market_price, S, K, T, r, Option_type)`

Calculates implied volatility using the least-squares method, supporting vectorized operations.

#### `newton_raphson_iv(market_price, S, K, T, r, Option_type, initial_guess=0.2, max_iterations=100, tolerance=1e-6)`

Computes implied volatility using the Newton-Raphson method.

#### Example

```python
from option_pricing import implied_volatility

# Calculate implied volatility for a call option
iv = implied_volatility(market_price=10, S=100, K=100, T=1, r=0.05, type='call')
print(f"Implied Volatility: {iv:.2%}")
```

### Plotting Volatility Surfaces

Visualize implied volatility surfaces using Plotly.

#### `plot_vol_surface(K_range, T_range, implied_vols, option_type='call')`

Plots the implied volatility surface for call or put options.

#### Example

```python
import numpy as np
from option_pricing import plot_vol_surface

# Define strike price and maturity ranges
K_range = np.linspace(80, 120, 40)
T_range = np.linspace(0.1, 1, 10)
implied_vols = np.random.rand(len(T_range), len(K_range))  # Dummy data

# Plot the implied volatility surface
plot_vol_surface(K_range, T_range, implied_vols, option_type='call')
```
## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any inquiries, please contact [wassimkerdoun5@gmail.com](mailto:wassimkerdoun5@gmail.com).

---
