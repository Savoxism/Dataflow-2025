from vnstock import Vnstock
stock = Vnstock().stock(symbol='CMG', source='VCI')

# print(stock.quote.history(start='2024-03-15', end='2024-06-23'))

# print(stock.finance.income_statement(period='quarter', lang='vi'))

# print(stock.finance.cash_flow(period='quarter', lang='vi'))

# print(stock.finance.ratio(period='quarter', lang='vi'))

df_finance = stock.quote.history(start='2025-01-01', end='2025-01-31')

df_finance.to_csv('CMG_2025_quote_history', index=False)

print('done!')
