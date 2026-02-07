
import pandas as pd
import plotly.express as px

BASE_PATH = r"C:\Users\user\Desktop\Talktech intern\processed"

tx_path = BASE_PATH + r"\transactions_clean.csv"
monthly_path = BASE_PATH + r"\monthly_agg_continuous.csv"
cat_month_path = BASE_PATH + r"\cat_month.csv"

#Monthly total expense trend
monthly = pd.read_csv(monthly_path, parse_dates=['month'])

fig = px.line(
    monthly,
    x='month',
    y='total_expense',
    title='Monthly Total Expense',
    labels={'total_expense': 'Total Expense', 'month': 'Month'}
)
fig.show()

#Spending by category (pie chart)
tx = pd.read_csv(tx_path, parse_dates=['date'])

cat_sum = (
    tx[tx['transaction_type'] == 'expense']
    .groupby('category_norm')['amount_abs']
    .sum()
    .reset_index()
)

fig = px.pie(
    cat_sum,
    names='category_norm',
    values='amount_abs',
    title='Spending Distribution by Category'
)
fig.show()

