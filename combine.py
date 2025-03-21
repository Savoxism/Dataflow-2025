import os
import pandas as pd
from vnstock import Vnstock

# symbols = ['CMG', 'CTG', 'CTS', 'FPT', 'HAG', 'ITD', 'KDH', 'MSN', 'MWG', 'NLG', 'NVB', 'PDR', 'POT', 'PVB', 'PVC', 'PVD', 'PVS', 'VCB', 'VIC', 'VNM']

symbols = ['NVB', 'PDR', 'POT', 'PVB', 'PVC', 'PVD', 'PVS', 'VCB', 'VIC', 'VNM']

start_date = pd.to_datetime("2013-03-31")
end_date = pd.to_datetime("2024-12-31")
quarter_end_map = {1: '-03-31', 2: '-06-30', 3: '-09-30', 4: '-12-31'}

# os.makedirs("files", exist_ok=True)

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