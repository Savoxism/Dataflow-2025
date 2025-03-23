from vnstock import Vnstock

symbol = 'FPT'
stock = Vnstock().stock(symbol=symbol, source='TCBS')

# print(stock.quote.history(start='2024-03-15', end='2024-06-23'))
# print(stock.finance.income_statement(period='quarter', lang='vi'))
# print(stock.finance.cash_flow(period='quarter', lang='vi'))
# print(stock.finance.ratio(period='quarter', lang='vi'))

df_quote_history= stock.quote.history(start='2018-01-01', end='2025-01-31')
# df_cash_flow = stock.finance.cash_flow(period='quarter') 
# df_income_statement = stock.finance.income_statement(period='quarter')
# df_ratio = stock.finance.ratio(period='quarter') 

df_quote_history.to_csv(f'new/{symbol}_quote_history.csv', index=False)
# df_cash_flow.to_csv(f'new/{symbol}_cash_flow.csv', index=False)
# df_income_statement.to_csv(f'new/{symbol}_income_statement.csv', index=False)
# df_ratio.to_csv(f'new/{symbol}_ratio.csv', index=False)
print('done!')

