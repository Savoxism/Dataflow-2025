import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the list of features to be used
features = [
    'Price',
    'Open',
    'High',
    'Low',
    'Vol.',
    'Change %',
    'return_day',
    'return_week',
    'return_month',
    'volatility_day',
    'volatility_week',
    'volatility_month',
    'liquidity_day',
    'liquidity_week',
    'liquidity_month',
    'z_score',
    'SMA_20',
    'SMA_50',
    'SMA_100',
    'WMA_20',
    'WMA_50',
    'WMA_100',
    'EMA26',
    'EMA12',
    'MACD',
    'Signal_Line',
    'RSI_14',
    'BB_upper',
    'BB_lower',
    'StochRSI_14',
    'Lãi/Lỗ ròng trước thuế',
    'Khấu hao TSCĐ',
    'Dự phòng RR tín dụng',
    'Lãi/Lỗ chênh lệch tỷ giá chưa thực hiện',
    'Lãi/Lỗ từ hoạt động đầu tư',
    'Thu nhập lãi',
    'Thu lãi và cổ tức',
    'Lưu chuyển tiền thuần từ HĐKD trước thay đổi VLĐ',
    'Tăng/Giảm các khoản phải thu',
    'Tăng/Giảm hàng tồn kho',
    'Tăng/Giảm các khoản phải trả',
    'Tăng/Giảm chi phí trả trước',
    'Chi phí lãi vay đã trả',
    'Tiền thu nhập doanh nghiệp đã trả',
    'Tiền thu khác từ các hoạt động kinh doanh',
    'Tiền chi khác từ các hoạt động kinh doanh',
    'Lưu chuyển tiền tệ ròng từ các hoạt động SXKD',
    'Mua sắm TSCĐ',
    'Tiền thu được từ thanh lý tài sản cố định',
    'Tiền chi cho vay, mua công cụ nợ của đơn vị khác (đồng)',
    'Tiền thu hồi cho vay, bán lại các công cụ nợ của đơn vị khác (đồng)',
    'Đầu tư vào các doanh nghiệp khác',
    'Tiền thu từ việc bán các khoản đầu tư vào doanh nghiệp khác',
    'Tiền thu cổ tức và lợi nhuận được chia',
    'Lưu chuyển từ hoạt động đầu tư',
    'Tăng vốn cổ phần từ góp vốn và/hoặc phát hành cổ phiếu',
    'Chi trả cho việc mua lại, trả cổ phiếu',
    'Tiền thu được các khoản đi vay',
    'Tiền trả các khoản đi vay',
    'Cổ tức đã trả',
    'Lưu chuyển tiền từ hoạt động tài chính',
    'Lưu chuyển tiền thuần trong kỳ',
    'Tiền và tương đương tiền',
    'Ảnh hưởng của chênh lệch tỷ giá',
    'Tiền và tương đương tiền cuối kỳ',
    'Tăng trưởng doanh thu (%)',
    'Doanh thu (đồng)',
    'Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)',
    'Tăng trưởng lợi nhuận (%)',
    'Thu nhập tài chính',
    'Chi phí tiền lãi vay',
    'Doanh thu bán hàng và cung cấp dịch vụ',
    'Các khoản giảm trừ doanh thu',
    'Doanh thu thuần',
    'Giá vốn hàng bán',
    'Lãi gộp',
    'Chi phí tài chính',
    'Lãi/lỗ từ công ty liên doanh',
    'Chi phí bán hàng',
    'Chi phí quản lý DN',
    'Lãi/Lỗ từ hoạt động kinh doanh',
    'Thu nhập khác',
    'Lãi lỗ trong công ty liên doanh, liên kết',
    'Thu nhập/Chi phí khác',
    'Lợi nhuận khác',
    'LN trước thuế',
    'Chi phí thuế TNDN hiện hành',
    'Chi phí thuế TNDN hoãn lại',
    'Lợi nhuận thuần',
    'Cổ đông thiểu số',
    'Cổ đông của Công ty mẹ',
    '(Vay NH+DH)/VCSH',
    'Nợ/VCSH',
    'TSCĐ / Vốn CSH',
    'Vốn CSH/Vốn điều lệ',
    'Vòng quay tài sản',
    'Vòng quay TSCĐ',
    'Số ngày thu tiền bình quân',
    'Số ngày tồn kho bình quân',
    'Số ngày thanh toán bình quân',
    'Chu kỳ tiền',
    'Vòng quay hàng tồn kho',
    'Biên EBIT (%)',
    'Biên lợi nhuận gộp (%)',
    'Biên lợi nhuận ròng (%)',
    'ROE (%)',
    'ROIC (%)',
    'ROA (%)',
    'EBITDA (Tỷ đồng)',
    'EBIT (Tỷ đồng)',
    'Tỷ suất cổ tức (%)',
    'Chỉ số thanh toán hiện thời',
    'Chỉ số thanh toán tiền mặt',
    'Chỉ số thanh toán nhanh',
    'Khả năng chi trả lãi vay',
    'Đòn bẩy tài chính',
    'Vốn hóa (Tỷ đồng)',
    'Số CP lưu hành (Triệu CP)',
    'P/E',
    'P/B',
    'P/S',
    'P/Cash Flow',
    'EPS (VND)',
    'BVPS (VND)',
    'EV/EBITDA'
]


window_size = 30
forecast_horizon = 30

def create_sequences(data, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size : i + window_size + forecast_horizon, 0])
    return np.array(X), np.array(y)

folder_path = "feature_engineered"
file_paths = glob.glob(os.path.join(folder_path, "feature_engineered_*.csv"))


results = []
for file_path in file_paths:
    base_name = os.path.basename(file_path)
    symbol = base_name.split("_")[-1].split(".")[0]
    
    df = pd.read_csv(file_path)
    data = df[features].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    

    X, y = create_sequences(data_scaled, window_size, forecast_horizon)
    print(f"{symbol}: X shape: {X.shape}, y shape: {y.shape}")
    
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(window_size, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    history = model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2, verbose=0)
    
    last_window = data_scaled[-window_size:]
    last_window = np.expand_dims(last_window, axis=0)  # shape: (1, window_size, num_features)
    predicted_scaled = model.predict(last_window)      # shape: (1, forecast_horizon)
    
    dummy = np.zeros((forecast_horizon, len(features) - 1))
    pred_full = np.concatenate([predicted_scaled.T, dummy], axis=1)
    predicted_prices = scaler.inverse_transform(pred_full)[:, 0]
    
    print(f"{symbol} predicted prices for next {forecast_horizon} days:")
    print(predicted_prices)
    
    # Append the predictions along with the symbol to the results list
    result_dict = {"Symbol": symbol}
    result_dict.update({f"Day_{i+1}": predicted_prices[i] for i in range(forecast_horizon)})
    results.append(result_dict)
    

df_results = pd.DataFrame(results)
output_file = "predicted_prices_30days.csv"
df_results.to_csv(output_file, index=False)
print(f"Saved combined predictions to {output_file}")