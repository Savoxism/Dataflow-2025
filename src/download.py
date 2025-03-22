from vnstock import Vnstock
stock = Vnstock().stock(symbol='VNI', source='VCI')

# print(stock.quote.history(start='2024-03-15', end='2024-06-23'))
# print(stock.finance.income_statement(period='quarter', lang='vi'))
# print(stock.finance.cash_flow(period='quarter', lang='vi'))
# print(stock.finance.ratio(period='quarter', lang='vi'))

df_quote_history= stock.quote.history(start='2025-01-01', end='2025-01-31')
df_cash_flow = stock.finance.cash_flow(period='quarter', lang='vi') 
df_income_statement = stock.finance.income_statement(period='quarter', lang='vi')
df_ratio = stock.finance.ratio(period='quarter', lang='vi')

df_quote_history.to_csv('files/CMG_2025_quote_history.csv', index=False)
df_cash_flow.to_csv('files/CMG_2025_cash_flow.csv', index=False)
df_income_statement.to_csv('files/CMG_2025_income_statement.csv', index=False)
df_ratio.to_csv('files/CMG_2025_ratio.csv', index=False)
print('done!')
