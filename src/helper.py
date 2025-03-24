import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.parser import parse
from scipy.optimize import minimize, Bounds
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt

def mean_returns(df, length): 
  mu = df.sum(axis = 0)/length
  return mu

def window_generator(dataframe, lookback, horizon, frequency, mode='fixed'):
    windows = []
    horizons = []
    
    df = dataframe.sort_index()
    start_date = df.index[0]
    end_date = df.index[-1]
    
    current_date = start_date

    if mode == 'fixed':
        while current_date + lookback + horizon <= end_date:
            window_start = current_date
            window_end = current_date + lookback
            horizon_start = window_end
            horizon_end = window_end + horizon
            
            windows.append(df.loc[window_start:window_end])
            horizons.append(df.loc[horizon_start:horizon_end])
            
            current_date = current_date + frequency

    elif mode == 'expanding':
        while current_date + horizon <= end_date:
            window_start = start_date
            window_end = current_date
            horizon_start = current_date
            horizon_end = current_date + horizon
            
            windows.append(df.loc[window_start:window_end])
            horizons.append(df.loc[horizon_start:horizon_end])
            
            current_date = current_date + frequency

    elif mode == 'adaptive':
        threshold = 0.05  # Set an arbitrary volatility threshold
        fixed_vol_window = pd.Timedelta(days=30)  # Use the last 30 days to measure volatility
        
        while current_date + horizon <= end_date:
            # Determine volatility over the fixed past window
            vol_window = df.loc[max(start_date, current_date - fixed_vol_window):current_date]
            volatility = vol_window.std().mean()  # average volatility across features
            
            # If volatility is high, use a shorter lookback; otherwise, use the default.
            if volatility > threshold:
                adaptive_lookback = pd.Timedelta(days=60)
            else:
                adaptive_lookback = lookback
            
            window_start = current_date - adaptive_lookback
            if window_start < start_date:
                window_start = start_date
            window_end = current_date
            horizon_start = current_date
            horizon_end = current_date + horizon
            
            windows.append(df.loc[window_start:window_end])
            horizons.append(df.loc[horizon_start:horizon_end])
            
            current_date = current_date + frequency
    else:
        raise ValueError("Unsupported mode. Use 'fixed', 'expanding', or 'adaptive'.")
    
    return windows, horizons


def actual_return(actual_returns, w):
  actual_returns = actual_returns
  mean_return = mean_returns(actual_returns, actual_returns.shape[0])
  actual_covariance = actual_returns.cov()

  portfolio_returns = mean_return.T.dot(w)
  portfolio_variance = w.T.dot(actual_covariance).dot(w)
  return portfolio_returns, portfolio_variance


def scipy_opt(predicted_returns, actual_returns, lam1, lam2):
  mean_return = mean_returns(predicted_returns, predicted_returns.shape[0])
  predicted_covariance = predicted_returns.cov()
  
  def f(w):
    return -(mean_return.T.dot(w) - lam1*(w.T.dot(predicted_covariance).dot(w)) + lam2*norm(w, ord=1))

  opt_bounds = Bounds(0, 1)

  def h(w):
    return sum(w) - 1

  #Constraints Dictionary
  cons = ({
      'type' : 'eq',
      'fun' : lambda w: h(w)
  })

  #Solver
  sol = minimize(f,
                 x0 = np.ones(mean_return.shape[0]),
                 constraints = cons,
                 bounds = opt_bounds,
                 options = {'disp': False},
                 tol=10e-10)

  w = sol.x
  predicted_portfolio_returns = w.dot(mean_return)
  portfolio_STD = w.T.dot(predicted_covariance).dot(w)
  
  portfolio_actual_returns, portfolio_actual_variance = actual_return(actual_returns, w)
    
  # Correct calculation using the square root of the variance to get the standard deviation.
  sharpe_ratio = portfolio_actual_returns / np.sqrt(portfolio_actual_variance)

  ret_dict = {'weights' : w,
              'predicted_returns' : predicted_portfolio_returns,
              'predicted_variance' : portfolio_STD,
              'actual_returns' : portfolio_actual_returns,
              'actual_variance' : portfolio_actual_variance,
              'sharpe_ratio': sharpe_ratio}
  
  return ret_dict

def metrics(returns): 
  sharpe = returns.mean() / returns.std()
  annualized_sharpe = sharpe.item() / math.sqrt(252)

  stdev = returns.std() 
  annualized_vol = stdev.item() / math.sqrt(252)

  return {"Annualized Sharpe Ratio": annualized_sharpe,
          "Annualized Volatility": annualized_vol}