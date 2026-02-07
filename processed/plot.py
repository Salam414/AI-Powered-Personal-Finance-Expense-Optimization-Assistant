import pandas as pd
import plotly.express as px
BASE = r"C:\Users\user\Desktop\Talktech intern\processed"
monthly = pd.read_csv(BASE + r"\monthly_agg_continuous.csv", parse_dates=['month'])
monthly = monthly.sort_values('month').reset_index(drop=True)

# check types and ranges
print(monthly[['total_income','total_expense']].dtypes)
print(monthly[['total_income','total_expense']].describe())

# show top spikes (helps spot outliers)
print("Top incomes\n", monthly.sort_values('total_income', ascending=False).head(5)[['month','total_income']])
print("Top expenses\n", monthly.sort_values('total_expense', ascending=False).head(5)[['month','total_expense']])

#Income vs Expense (monthly)
df = monthly.copy()
df['expense_neg'] = -df['total_expense']   # flip expense sign for plotting

df_long = df.melt(id_vars='month', value_vars=['total_income','expense_neg'],
                  var_name='variable', value_name='amount')
# rename display labels
df_long['variable'] = df_long['variable'].map({'total_income':'Income','expense_neg':'Expense'})

fig = px.bar(df_long, x='month', y='amount', color='variable',
             title='Income (up) vs Expense (down) per Month')
fig.update_layout(barmode='relative')
fig.update_yaxes(title_text='Amount')
fig.show()
