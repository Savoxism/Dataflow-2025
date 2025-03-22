import os
import pandas as pd

df = pd.read_csv("processed_stock_data.csv")

output_folder = "processed_stck_data"
os.makedirs(output_folder, exist_ok=True)

symbols = df["Symbol"].unique()

for symbol in symbols:
    df_symbol = df[df["Symbol"] == symbol]
    file_path = os.path.join(output_folder, f"{symbol}.csv")
    df_symbol.to_csv(file_path, index=False)
    print(f"Đã lưu file: {file_path}")

