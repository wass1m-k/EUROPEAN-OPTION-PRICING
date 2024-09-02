import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
import plotly.graph_objects as go
import plotly.io as pio
from typing import Literal
import math


class EuropeanOption:
    
    def __init__(self,S,K,T,r,sigma,Option_type:Literal['call','put']) -> None:
        
        """
        Initiate an Instance of the EuropeanOption class.

        ## Parameters:
        - S: Current price of the underlying asset (e.g., stock)
        - K: Strike price of the option
        - T: Time to expiration (in years)
        - r: Risk-free interest rate (continuously compounded)
        - sigma: Volatility of the underlying asset (standard deviation of the asset's returns)
        - type: Option type (Call or Put)
        """   
        self._S = S
        self._K = K
        self._T = T
        self._r = r
        self._sigma = sigma
        self._Option_type = Option_type
        self._price = self.black_and_scholes()

    def black_and_scholes(self):
        
        """
        Calculate the price of European options.

        ## Parameters:
        - self: Instance of the EuropeanOption.

        ## Returns:
        - float: The theoretical price of the European option according to a Black, Scholes & Merton model.
        """
        
        d1, d2 = self._d1_d2()
        
        N = norm.cdf
        
        if self._Option_type.lower() == 'call':
            price = N(d1)*self._S - N(d2)*self._K*np.exp(-self._r*self._T)
        
        elif self._Option_type.lower() == 'put':
            price = N(-d2)*self._K*np.exp(-self._r*self._T) - self._S*N(-d1)
            
        else:
            raise Exception('Please enter "Call" or "Put" in Option type.')
        
        return price
    
    def merton_jump_diffusion(self, mu_j, sigma_j, lam, max_iter=100, stop_cond=1e-15):
        """
        Calculate the price of European options using the Merton jump diffusion model.

        Parameters:
        - mu_j: Mean of the jump size
        - sigma_j: Volatility of the jump size
        - lam: Jump intensity (i.e., the expected number of jumps per year)
        - max_iter: Maximum number of iterations (default 100)
        - stop_cond: Stop condition for the series expansion (default 1e-15)

        Returns:
        - float: The theoretical price of the European option according to the Merton jump diffusion model.
        """
        
        V = 0
        
        for k in range(max_iter):
            sigma_k = np.sqrt(self._sigma**2 + k * sigma_j**2 / self._T)
            r_k = self._r - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1) + (k * (mu_j + 0.5 * sigma_j**2)) / self._T
            poisson_weight = (np.exp(-lam * self._T) * (lam * self._T)**k / math.factorial(k))
            
            bs_value = EuropeanOption(self._S, self._K, self._T, r_k, sigma_k, self._Option_type).black_and_scholes()
            sum_k = poisson_weight * bs_value
            V += sum_k
            
            if sum_k < stop_cond:  # if the last added component is below the threshold, return the current sum
                return V
        
        return V
        
    def binomial_tree(self,n):
        """
        Calculate the price of European options using the Binomial Tree model.
        
        ## Parameters:
        - n: Number of steps in the binomial tree.

        ## Returns:
        - float: The theoretical price of the European option according to the Binomial Tree model.
        """
        # Up and Down Factors
        u = np.exp(self._sigma * np.sqrt(self._T / n))
        d = 1 / u
        # Risk-neutral probability
        p = (np.exp(self._r * self._T / n) - d) / (u - d)
        
        asset_prices = np.zeros(n+1)
        option_prices = np.zeros(n+1)
        
        for i in range(n+1):
            asset_prices[i] = self._S * (u ** (n-i)) * (d ** i)
            if self._Option_type.lower() == 'call':
                option_prices[i] = max(asset_prices[i] - self._K,0)
            elif self._Option_type.lower() == 'put':
                option_prices[i] = max(self._K - asset_prices[i],0)
            else:
                raise ValueError("Option type can only be 'Call' or 'Put'.")
        
        # Work Backwards throught the tree
        for j in range(n-1,-1,-1):
            for i in range(j+1):
                option_prices[i] = (p*option_prices[i] + (1-p) * option_prices[i+1]) * np.exp(-self._r * self._T/n)
        
        return option_prices[0]
            
    def _d1_d2(self):
        
        d1 = (np.log(self._S/self._K) + (self._r + (self._sigma**2)/2)*self._T)/(self._sigma*np.sqrt(self._T))
        d2 = d1 - self._sigma*np.sqrt(self._T)
        
        return d1,d2
    
    def delta(self):
        
        N = norm.cdf
        d1,_ = self._d1_d2()
        
        if self._Option_type == 'call':
            return N(d1)
        
        elif self._Option_type == 'put':
            return N(d1)-1
        
    def gamma(self):
        
        n = norm.pdf
        d1,_ = self._d1_d2()
        
        return n(d1) / (self._S * self._sigma * np.sqrt(self._T))
    
    def vega(self):
        
        n = norm.pdf
        d1,_ = self._d1_d2()
        
        return self._S*n(d1)*np.sqrt(self._T)
    
    def theta(self):
        
        N = norm.cdf
        n = norm.pdf
        d1, d2 = self._d1_d2()
        
        if self._Option_type == 'call':
            theta = (-self._S * n(d1) * self._sigma / (2 * np.sqrt(self._T))
                    - self._r * self._K * np.exp(-self._r * self._T) * N(d2))
        
        elif self._Option_type == 'put':
            theta = (-self._S * n(d1) * self._sigma / (2 * np.sqrt(self._T))
                    + self._r * self._K * np.exp(-self._r * self._T) * N(-d2))
        
        
        return theta
        
    def rho(self):
        
        N = norm.cdf
        _,d2 = self._d1_d2()
        
        if self._Option_type == 'call':
            return self._K * self._T * np.exp(-self._r * self._T) * N(d2)
        
        elif self._Option_type == 'put':
            return -self._K * self._T * np.exp(-self._r * self._T) * N(-d2)
        
        
    @property
    def S(self):
        return self._S

    @property
    def K(self):
        return self._K

    @property
    def T(self):
        return self._T

    @property
    def r(self):
        return self._r

    @property
    def sigma(self):
        return self._sigma

    @property
    def type(self):
        return self._Option_type

    @property
    def price(self):
        return self._price
    
    @S.setter
    def S(self,value) -> None:
        self._S = value
        self._price = self.black_and_scholes()
        
    @K.setter
    def K(self,value) -> None:
        self._K = value
        self._price = self.black_and_scholes()
        
    @T.setter
    def T(self,value) -> None:
        self._T = value
        self._price = self.black_and_scholes()
      
    @r.setter
    def r(self,value) -> None:
        self._r = value
        self._price = self.black_and_scholes()
    
    @sigma.setter
    def sigma(self,value) -> None:
        self._sigma = value
        self._price = self.black_and_scholes()
        
    @type.setter
    def  type(self,value: Literal['call','put']) -> None:
        
        if value.lower() not in ['call','put']:
            raise ValueError("Option type can only be Call or Put.")
        
        self._Option_type = value.lower()
        self._price = self.black_and_scholes()
        
    def __repr__(self) -> str:
        return f"""European {self._Option_type.lower()} option | S = ${self._S} | K = ${self._K} | T = {self._T} {'year' if self._T==1 else 'years'} | r = {self._r*100}% | C = ${self._price:.2f}"""

def implied_volatility(market_price, S, K, T, r, type: Literal['call', 'put']):
        """
        Calculate the implied volatility of an European option using the market price.

        The implied volatility is the volatility value that, when input into the Black-Scholes model, 
        yields the market price of the option. This function uses numerical optimization to find the 
        implied volatility by minimizing the difference between the market price and the option price 
        calculated by the Black-Scholes model.

        ## Parameters:
        - market_price (float): The market price of the European option.
        - S0 (float): Current price of the underlying asset (e.g., stock).
        - K (float): Strike price of the option.
        - T (float): Time to expiration (in years).
        - r (float): Risk-free interest rate (continuously compounded).
        - type (Literal['call', 'put']): Option type ('call' or 'put').

        ## Returns:
        - float: The implied volatility of the European option.

        ## Method:
        The function defines an objective function that calculates the absolute difference between the 
        market price and the price of the European option calculated using the Black-Scholes formula. 
        The `least_squares` function from `scipy.optimize` is then used to minimize this difference by 
        varying the volatility. The optimized volatility is returned as the implied volatility.
        """
        
        obj = lambda sigma: np.abs(market_price - EuropeanOption(S, K, T, r, sigma, type).price)
        return least_squares(obj, [0.1]).x[0]


def least_squares_iv(market_price, S, K, T, r, Option_type: Literal['call', 'put']):
    """
    Calculate the implied volatility of European options using the least-squares method.

    Parameters:
    - market_price (float or np.ndarray): Market price of the European option(s).
    - S (float or np.ndarray): Current price of the underlying asset.
    - K (float or np.ndarray): Strike price of the option(s).
    - T (float or np.ndarray): Time to expiration (in years).
    - r (float): Risk-free interest rate (continuously compounded).
    - Option_type (Literal['call', 'put']): Option type ('call' or 'put').

    Returns:
    - float or np.ndarray: The implied volatility(ies) of the European option(s).
    """
    
    def calculate_iv(market_price, S, K, T, r, Option_type):
        obj = lambda sigma: np.abs(market_price - EuropeanOption(S, K, T, r, sigma, Option_type).price)
        
        result = least_squares(obj, [0.1])
        return result.x[0]

    # Vectorize the function to handle arrays
    vectorized_iv = np.vectorize(calculate_iv)
    
    return vectorized_iv(market_price, S, K, T, r, Option_type)


def newton_raphson_iv(market_price, S, K, T, r, Option_type: Literal['call', 'put'],
                       initial_guess=0.2, max_iterations=100, tolerance=1e-6):
    """
    Implements the Newton-Raphson method to calculate implied volatility.

    Parameters:
    - market_price: Market price of the European option (can be an array).
    - S: Current price of the underlying asset.
    - K: Strike price of the option (can be an array).
    - T: Time to expiration (in years).
    - r: Risk-free interest rate (continuously compounded).
    - Option_type: Option type ('call' or 'put').
    - initial_guess: Initial guess for implied volatility.
    - max_iterations: Maximum number of iterations.
    - tolerance: Absolute tolerance for convergence.

    Returns:
    - np.array: Implied volatility of the European option.
    """

    sigma = np.full_like(market_price, initial_guess, dtype=np.float64)

    for iteration in range(max_iterations):
        option = EuropeanOption(S, K, T, r, sigma, Option_type)
        price = option.price
        vega = option.vega()

        delta_price = market_price - price

        if (np.abs(delta_price) < tolerance).all():
            return sigma

        if (np.abs(vega) < 1e-8).any():
            print("Warning: Small vega encountered. Consider adjusting initial guess or using alternative method.")

        sigma_update = delta_price / vega
        sigma = sigma + 0.5 * sigma_update

    print("Newton-Raphson method did not converge: returning sigma from last iteration")
    return sigma


def plot_vol_surface(K_range, T_range, implied_vols, option_type='call'):
    """
    Plots the implied volatility surface for either call or put options.

    Parameters:
    - K_range: Array-like, range of strike prices.
    - T_range: Array-like, range of maturities.
    - implied_vols: 2D array of implied volatilities.
    - option_type: String, 'call' or 'put', determines the plot title and colorscale.
    """
    K_grid, T_grid = np.meshgrid(K_range, T_range)

    if option_type == 'call':
        colorscale = 'Viridis'
        title = 'Call Implied Volatility Surface'
    elif option_type == 'put':
        colorscale = 'Cividis'
        title = 'Put Implied Volatility Surface'
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    surface = go.Surface(
        x=K_grid,
        y=T_grid,
        z=implied_vols.T,  # Transpose to align with the grid
        colorscale=colorscale,
        colorbar=None,
        opacity=0.9
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='Strike Price (K)'),
            yaxis=dict(title='Maturity (T)'),
            zaxis=dict(title='Implied Volatility'),
        ),
        autosize=True,
        width=1200,
        height=800,
    )

    fig = go.Figure(data=[surface], layout=layout)
    pio.show(fig)