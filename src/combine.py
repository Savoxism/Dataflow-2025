import os
import pandas as pd
from vnstock import Vnstock

# symbols = ['CMG', 'CTG', 'CTS', 'FPT', 'HAG', 'ITD', 'KDH', 'MSN', 'MWG', 'NLG', 'NVB']
# symbols = ['HAG', 'ITD', 'KDH', 'MSN', 'MWG', 'NLG', 'NVB']
# symbols = ['NVB', 'PDR', 'POT', 'PVB', 'PVC', 'PVD', 'PVS', 'VCB', 'VIC', 'VNM']

symbols = ['CMG']

rename_dict = {
    'Date': 'date',
    'Close': 'close',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Volume': 'volume',
    'Change': 'change',
    'Symbol': 'symbol',
    'return_day': 'return_day',
    'return_week': 'return_week',
    'return_month': 'return_month',
    'volatility_day': 'volatility_day',
    'volatility_week': 'volatility_week',
    'volatility_month': 'volatility_month',
    'liquidity_day': 'liquidity_day',
    'liquidity_week': 'liquidity_week',
    'liquidity_month': 'liquidity_month',
    'high_minus_close': 'high_minus_close',
    'low_minus_open': 'low_minus_open',
    'cumulative_return': 'cumulative_return',
    'Stochastic_Osc': 'stochastic_osc',
    'ATR': 'atr',
    'ADX14': 'adx_14',
    'ADX20': 'adx_20',
    'SMA_3': 'sma_3',
    'SMA_7': 'sma_7',
    'SMA_14': 'sma_14',
    'SMA_21': 'sma_21',
    'SMA_50': 'sma_50',
    'SMA_100': 'sma_100',
    'WMA_3': 'wma_3',
    'WMA_7': 'wma_7',
    'WMA_14': 'wma_14',
    'WMA_21': 'wma_21',
    'WMA_50': 'wma_50',
    'WMA_100': 'wma_100',
    'EMA6': 'ema_6',
    'EMA12': 'ema_12',
    'EMA26': 'ema_26',
    'outMACD': 'out_macd',
    'outMACDSignal': 'out_macd_signal',
    'outMACDHist': 'out_macd_hist',
    'RSI6': 'rsi_6',
    'RSI12': 'rsi_12',
    'RSI14': 'rsi_14',
    'StochRSI_6': 'stochrsi_6',
    'StochRSI_12': 'stochrsi_12',
    'StochRSI_14': 'stochrsi_14',
    'BBANDSMIDDLE': 'bbands_middle',
    'BBANDSUPPER': 'bbands_upper',
    'BBANDSLOWER': 'bbands_lower',
    'OBV': 'obv',
    'MFI14': 'mfi_14',
    'MOM1': 'mom_1',
    'MOM3': 'mom_3',
    'MOM7': 'mom_7',
    'CCI12': 'cci_12',
    'CCI20': 'cci_20',
    'ROCR3': 'rocr_3',
    'ROCR12': 'rocr_12',
    'WILLR': 'willr',
    'TRIX': 'trix',
    'Lãi/Lỗ ròng trước thuế': 'net_profit_loss_before_tax',
    'Khấu hao TSCĐ': 'depreciation_fixed_assets',
    'Dự phòng RR tín dụng': 'credit_risk_reserve',
    'Lãi/Lỗ chênh lệch tỷ giá chưa thực hiện': 'unrealized_forex_gain_loss',
    'Lãi/Lỗ từ hoạt động đầu tư': 'investment_income_loss',
    'Thu nhập lãi': 'interest_income',
    'Thu lãi và cổ tức': 'interest_dividend_income',
    'Lưu chuyển tiền thuần từ HĐKD trước thay đổi VLĐ': 'net_cash_flow_operating_before_wc_changes',
    'Tăng/Giảm các khoản phải thu': 'change_in_receivables',
    'Tăng/Giảm hàng tồn kho': 'change_in_inventories',
    'Tăng/Giảm các khoản phải trả': 'change_in_payables',
    'Tăng/Giảm chi phí trả trước': 'change_in_prepaid_expenses',
    'Chi phí lãi vay đã trả': 'interest_expense_paid',
    'Tiền thu nhập doanh nghiệp đã trả': 'corporate_income_tax_paid',
    'Tiền thu khác từ các hoạt động kinh doanh': 'other_cash_inflows_operating',
    'Tiền chi khác từ các hoạt động kinh doanh': 'other_cash_outflows_operating',
    'Lưu chuyển tiền tệ ròng từ các hoạt động SXKD': 'net_cash_flow_from_business_activities',
    'Mua sắm TSCĐ': 'capital_expenditure',
    'Tiền thu được từ thanh lý tài sản cố định': 'cash_from_fixed_assets_sale',
    'Tiền chi cho vay, mua công cụ nợ của đơn vị khác (đồng)': 'cash_outflow_loans_debt_instruments',
    'Tiền thu hồi cho vay, bán lại các công cụ nợ của đơn vị khác (đồng)': 'cash_inflow_loan_recoveries',
    'Đầu tư vào các doanh nghiệp khác': 'investments_in_other_businesses',
    'Tiền thu từ việc bán các khoản đầu tư vào doanh nghiệp khác': 'cash_from_sale_investments',
    'Tiền thu cổ tức và lợi nhuận được chia': 'dividends_profit_received',
    'Lưu chuyển từ hoạt động đầu tư': 'cash_flow_investing',
    'Tăng vốn cổ phần từ góp vốn và/hoặc phát hành cổ phiếu': 'increase_in_equity',
    'Chi trả cho việc mua lại, trả cổ phiếu': 'cash_used_share_repurchase',
    'Tiền thu được các khoản đi vay': 'cash_from_borrowings',
    'Tiền trả các khoản đi vay': 'cash_repayment_borrowings',
    'Cổ tức đã trả': 'dividends_paid',
    'Lưu chuyển tiền từ hoạt động tài chính': 'cash_flow_financing',
    'Lưu chuyển tiền thuần trong kỳ': 'net_cash_flow_period',
    'Tiền và tương đương tiền': 'cash_and_equivalents',
    'Ảnh hưởng của chênh lệch tỷ giá': 'exchange_rate_effect',
    'Tiền và tương đương tiền cuối kỳ': 'ending_cash_equivalents',
    'Tăng trưởng doanh thu (%)': 'revenue_growth_percent',
    'Doanh thu (đồng)': 'revenue_vnd',
    'Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)': 'net_profit_after_tax_parent_vnd',
    'Tăng trưởng lợi nhuận (%)': 'profit_growth_percent',
    'Thu nhập tài chính': 'financial_income',
    'Chi phí tiền lãi vay': 'interest_expense',
    'Doanh thu bán hàng và cung cấp dịch vụ': 'sales_revenue_service_income',
    'Các khoản giảm trừ doanh thu': 'revenue_deductions',
    'Doanh thu thuần': 'net_revenue',
    'Giá vốn hàng bán': 'cogs',
    'Lãi gộp': 'gross_profit',
    'Chi phí tài chính': 'financial_expenses',
    'Lãi/lỗ từ công ty liên doanh': 'profit_loss_associates',
    'Chi phí bán hàng': 'selling_expenses',
    'Chi phí quản lý DN': 'administrative_expenses',
    'Lãi/Lỗ từ hoạt động kinh doanh': 'operating_profit_loss',
    'Thu nhập khác': 'other_income',
    'Lãi lỗ trong công ty liên doanh, liên kết': 'profit_loss_joint_ventures',
    'Thu nhập/Chi phí khác': 'other_income_expense',
    'Lợi nhuận khác': 'other_profit',
    'LN trước thuế': 'profit_before_tax',
    'Chi phí thuế TNDN hiện hành': 'current_corporate_tax_expense',
    'Chi phí thuế TNDN hoãn lại': 'deferred_corporate_tax_expense',
    'Lợi nhuận thuần': 'net_profit',
    'Cổ đông thiểu số': 'minority_interest',
    'Cổ đông của Công ty mẹ': 'parent_company_shareholders',
    '(Vay NH+DH)/VCSH': 'loans_to_equity_ratio',
    'Nợ/VCSH': 'debt_to_equity',
    'TSCĐ / Vốn CSH': 'fixed_assets_to_equity',
    'Vốn CSH/Vốn điều lệ': 'equity_to_chartered_capital',
    'Vòng quay tài sản': 'asset_turnover',
    'Vòng quay TSCĐ': 'fixed_asset_turnover',
    'Số ngày thu tiền bình quân': 'avg_collection_period',
    'Số ngày tồn kho bình quân': 'avg_inventory_days',
    'Số ngày thanh toán bình quân': 'avg_payment_days',
    'Chu kỳ tiền': 'cash_cycle',
    'Vòng quay hàng tồn kho': 'inventory_turnover',
    'Biên EBIT (%)': 'ebit_margin_percent',
    'Biên lợi nhuận gộp (%)': 'gross_profit_margin_percent',
    'Biên lợi nhuận ròng (%)': 'net_profit_margin_percent',
    'ROE (%)': 'roe_percent',
    'ROIC (%)': 'roic_percent',
    'ROA (%)': 'roa_percent',
    'EBITDA (Tỷ đồng)': 'ebitda_billion_vnd',
    'EBIT (Tỷ đồng)': 'ebit_billion_vnd',
    'Tỷ suất cổ tức (%)': 'dividend_yield_percent',
    'Chỉ số thanh toán hiện thời': 'current_ratio',
    'Chỉ số thanh toán tiền mặt': 'cash_ratio',
    'Chỉ số thanh toán nhanh': 'quick_ratio',
    'Khả năng chi trả lãi vay': 'interest_coverage',
    'Đòn bẩy tài chính': 'financial_leverage',
    'Vốn hóa (Tỷ đồng)': 'market_cap_billion_vnd',
    'Số CP lưu hành (Triệu CP)': 'shares_outstanding_million',
    'P/E': 'pe',
    'P/B': 'pb',
    'P/S': 'ps',
    'P/Cash Flow': 'p_cash_flow',
    'EPS (VND)': 'eps_vnd',
    'BVPS (VND)': 'bvps_vnd',
    'EV/EBITDA': 'ev_ebitda'
}

start_date = pd.to_datetime("2018-03-31")
end_date = pd.to_datetime("2024-12-31")
quarter_end_map = {1: '-03-31', 2: '-06-30', 3: '-09-30', 4: '-12-31'}

def load_and_process_quarterly(file_path, skip_first_two_lines=False):
    if skip_first_two_lines:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # bỏ dòng 1
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    df = pd.read_csv(file_path)
    if not {"Năm", "Kỳ"}.issubset(df.columns):
        raise ValueError(f"File {file_path} không có 'Năm' và 'Kỳ'")
    df["Quarter_End"] = pd.to_datetime(df["Năm"].astype(str) + df["Kỳ"].map(quarter_end_map))
    df = df[(df["Quarter_End"] >= start_date) & (df["Quarter_End"] <= end_date)]
    df.sort_values("Quarter_End", inplace=True)
    return df

def process_symbol(symbol):
    try:
        print(f"📥 Đang xử lý: {symbol}")
        stock = Vnstock().stock(symbol=symbol, source='VCI')

        df_cash = stock.finance.cash_flow(period='quarter', lang='vi')
        df_fin = stock.finance.income_statement(period='quarter', lang='vi')
        df_ratio = stock.finance.ratio(period='quarter', lang='vi')

        df_cash.to_csv(f"{symbol}_cash_flow.csv", index=False)
        df_fin.to_csv(f"{symbol}_financial_reports.csv", index=False)
        df_ratio.to_csv(f"{symbol}_stock_ratio.csv", index=False)

        daily_file = f"processed_stck_data/{symbol}.csv"
        df_daily = pd.read_csv(daily_file, parse_dates=["Date"])
        df_daily = df_daily[(df_daily["Date"] >= start_date) & (df_daily["Date"] <= end_date)]
        df_daily.sort_values("Date", inplace=True)

        df_cash = load_and_process_quarterly(f"{symbol}_cash_flow.csv")
        df_fin = load_and_process_quarterly(f"{symbol}_financial_reports.csv")
        df_ratio = load_and_process_quarterly(f"{symbol}_stock_ratio.csv", skip_first_two_lines=True)

        df_merged = pd.merge_asof(
            df_daily, df_cash, left_on="Date", right_on="Quarter_End", direction="backward", suffixes=("", "_cf")
        )
        df_merged = pd.merge_asof(
            df_merged, df_fin, left_on="Date", right_on="Quarter_End", direction="backward", suffixes=("", "_fr")
        )
        df_merged = pd.merge_asof(
            df_merged, df_ratio, left_on="Date", right_on="Quarter_End", direction="backward", suffixes=("", "_sr")
        )

        cols_to_drop = ["CP_sr", "Năm_sr", "Kỳ_sr", "Quarter_End_sr", "Quarter_End_fr",
                        "Quarter_End", "CP_fr", "Năm_fr", "Kỳ_fr", "CP", "Năm", "Kỳ"]
        df_merged.drop(columns=[col for col in cols_to_drop if col in df_merged.columns], inplace=True)
        
        # Rename the columns into English with underscores using the dictionary
        df_merged = df_merged.rename(columns=rename_dict)

        output_file = f"feature_engineered/feature_engineered_{symbol}.csv"
        df_merged.to_csv(output_file, index=False)
        print(f"✅ Hoàn tất: {output_file}")

    except Exception as e:
        print(f"❌ Lỗi với {symbol}: {e}")
        
for symbol in symbols:
    process_symbol(symbol)
    
# Xóa các file tạm sau khi hoàn tất
for symbol in symbols:
    try:
        os.remove(f"{symbol}_cash_flow.csv")
        os.remove(f"{symbol}_financial_reports.csv")
        os.remove(f"{symbol}_stock_ratio.csv")
    except Exception as e:
        print(f"❌ Không thể xóa file tạm cho {symbol}: {e}")
        
print("🗑️ Đã xóa các file tạm.")