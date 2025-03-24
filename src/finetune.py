import pandas as pd
import numpy as np
import math
from helper import *

def process_model(model_name, horizon_days):
    predicted_file = f'stage2_data/predicted_prices/{model_name}_{horizon_days}_Days_Predicted_Prices.csv'
    actual_file = f'stage2_data/actual_prices/{model_name}_{horizon_days}_Days_Actual_Prices.csv'
    
    predicted_prices = pd.read_csv(predicted_file)
    predicted_prices['Date'] = pd.to_datetime(predicted_prices['Date'])
    predicted_prices.set_index('Date', inplace=True)
    
    actual_prices = pd.read_csv(actual_file)
    actual_prices['Date'] = pd.to_datetime(actual_prices['Date'])
    actual_prices.set_index('Date', inplace=True)
    
    predicted_returns = predicted_prices.apply(lambda x: np.log(x) - np.log(x.shift(horizon_days))).iloc[horizon_days:]
    actual_returns = actual_prices.apply(lambda x: np.log(x) - np.log(x.shift(horizon_days))).iloc[horizon_days:]
    
    lookback_offset = pd.DateOffset(days=150)  
    horizon_offset  = pd.DateOffset(days=horizon_days)
    frequency_offset = pd.DateOffset(weeks=1)
    
    pred_windows, pred_horizons = window_generator(predicted_returns, lookback_offset, horizon_offset, frequency_offset, mode='fixed')
    act_windows, act_horizons = window_generator(actual_returns, lookback_offset, horizon_offset, frequency_offset, mode='fixed')
    
    actual_returns_list = []
    actual_variance_list = []
    sharpe_list = []
    equity = [100]
    
    # Lặp qua các cửa sổ (ví dụ: mỗi cửa sổ tính toán tối ưu một lần)
    for i in range(len(act_horizons)):
        result = scipy_opt(pred_horizons[i], act_horizons[i], lam1=0.5, lam2=2)
        actual_returns_list.append(result['actual_returns'])
        actual_variance_list.append(result['actual_variance'])
        sharpe_list.append(result['sharpe_ratio'])
        
        if i > 0:
            equity.append(equity[-1] * math.exp(result['actual_returns']))
    
    returns_array = np.array(actual_returns_list)
    metrics_dict = metrics(returns_array)
    
    return {
        'annualized_sharpe': metrics_dict["Annualized Sharpe Ratio"],
        'annualized_volatility': metrics_dict["Annualized Volatility"],
        'ending_equity': equity[-1]
    }

# Danh sách các mô hình và các khoảng thời gian cần so sánh
models = ['LSTM', 'GRU', 'CNN_BiLSTM']
horizons = [30, 60, 90, 180]

# results_list = []
# for model in models:
#     for h in horizons:
#         res = process_model(model, h)
#         res['model'] = model
#         res['horizon'] = h
#         results_list.append(res)

# results_df = pd.DataFrame(results_list, columns=['model', 'horizon', 'annualized_sharpe', 'annualized_volatility', 'ending_equity'])
# print(results_df)

# results_df.to_csv("final_result/final_results.csv", index=False)

def get_weights_for_model(model_name, horizon_days):
    predicted_file = f'stage2_data_150/predicted_prices/{model_name}_{horizon_days}_Days_Predicted_Prices.csv'
    actual_file = f'stage2_data_150/actual_prices/{model_name}_{horizon_days}_Days_Actual_Prices.csv'
    
    predicted_prices = pd.read_csv(predicted_file)
    predicted_prices['Date'] = pd.to_datetime(predicted_prices['Date'])
    predicted_prices.set_index('Date', inplace=True)
    
    actual_prices = pd.read_csv(actual_file)
    actual_prices['Date'] = pd.to_datetime(actual_prices['Date'])
    actual_prices.set_index('Date', inplace=True)
    
    predicted_returns = predicted_prices.apply(lambda x: np.log(x) - np.log(x.shift(horizon_days))).iloc[horizon_days:]
    actual_returns = actual_prices.apply(lambda x: np.log(x) - np.log(x.shift(horizon_days))).iloc[horizon_days:]
    
    lookback_offset = pd.DateOffset(days=150)  
    horizon_offset  = pd.DateOffset(days=horizon_days)
    frequency_offset = pd.DateOffset(weeks=1)
    
    pred_windows, pred_horizons = window_generator(predicted_returns, lookback_offset, horizon_offset, frequency_offset, mode='fixed')
    act_windows, act_horizons = window_generator(actual_returns, lookback_offset, horizon_offset, frequency_offset, mode='fixed')
    
    # Dùng cửa sổ cuối cùng để lấy trọng số danh mục
    result = scipy_opt(pred_horizons[-1], act_horizons[-1], lam1=0.5, lam2=2)
    
    weights = result['weights']
    stock_names = predicted_returns.columns.tolist()
    weight_dict = dict(zip(stock_names, weights))
    
    return weight_dict

models = ['LSTM', 'GRU', 'CNN_BiLSTM']
horizon = 60

for model in models:
    print(f"\nTrọng số danh mục tối ưu cho mô hình {model} (horizon={horizon} ngày):")
    weights = get_weights_for_model(model, horizon)
    for stock, w in weights.items():
        print(f"{stock}: {w:.2%}")
