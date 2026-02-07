import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Tuple, Dict, Any


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names for predictable access."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _guess_columns(df: pd.DataFrame) -> Dict[str,str]:
    """
    Try to find common alternative names for key columns and map them to
    canonical names used in the pipeline.
    """
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    # date
    for candidate in ['date', 'timestamp', 'tx_date', 'transaction_date', 'time']:
        if candidate in cols:
            mapping['date'] = cols[candidate]
            break
    # amount
    for candidate in ['amount', 'amt', 'value', 'transaction_amount']:
        if candidate in cols:
            mapping['amount'] = cols[candidate]
            break
    # description
    for candidate in ['description', 'memo', 'narration', 'details']:
        if candidate in cols:
            mapping['description'] = cols[candidate]
            break
    # merchant
    for candidate in ['merchant', 'vendor', 'payee', 'merchant_name']:
        if candidate in cols:
            mapping['merchant'] = cols[candidate]
            break
    # category
    for candidate in ['category', 'cat', 'label']:
        if candidate in cols:
            mapping['category'] = cols[candidate]
            break
    # balance
    for candidate in ['balance', 'running_balance', 'acct_balance']:
        if candidate in cols:
            mapping['balance'] = cols[candidate]
            break
    # account id
    for candidate in ['account_id', 'account', 'acct_id', 'acct']:
        if candidate in cols:
            mapping['account_id'] = cols[candidate]
            break

    return mapping

def _clean_text_basic(s: str) -> str:
    """Lightweight text cleaning for descriptions/merchant names (analytics focus)."""
    if pd.isna(s):
        return 'unknown'
    s = str(s).lower()
    # replace emails, phones, and long numeric sequences with tokens
    s = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' <EMAIL> ', s)
    s = re.sub(r'\+?\d[\d\-\s\(\)]{6,}\d', ' <PHONE> ', s)
    s = re.sub(r'\d{6,}', ' <NUM> ', s)    # long numbers (card/account)
    # drop punctuation
    s = re.sub(r'[^\w\s]', ' ', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    if s == '':
        return 'unknown'
    return s

# -------------------------
# Main preprocessing function
# -------------------------
def preprocess_core_transactions(
    csv_path: str,
    out_dir: str = 'data/processed',
    tz: str | None = None,
    drop_zero_amounts: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str,Any]]:
    """
    Preprocess a personal finance CSV for analytics / visualization / budgeting.

    Returns:
      cleaned_df, monthly_df (continuous monthly series), category_month_df, preprocessing_log

    Artifacts are saved under out_dir:
      - transactions_clean.csv
      - monthly_agg_continuous.csv
      - cat_month.csv
      - preprocessing_log.json
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log: Dict[str,Any] = {}

    # 1) Load and inspect
    df = pd.read_csv(csv_path, low_memory=False)
    log['initial_rows'] = len(df)
    log['initial_columns'] = list(df.columns)

    # 2) Standardize column names (lower case)
    df = _standardize_column_names(df)
    col_map = _guess_columns(df)
    log['guessed_columns'] = col_map.copy()

    # make sure canonical columns exist (create if missing)
    # We will refer to canonical names: date, amount, description, merchant, category, balance, account_id
    if 'date' not in col_map:
        # create placeholder if no date exists (rare)
        raise ValueError("No date-like column found in CSV. Preprocessing requires a date/timestamp column.")
    # rename df columns to canonical short names for pipeline convenience
    rename_map = {col_map[k]: k for k in col_map}
    df = df.rename(columns=rename_map)

    # 3) Parse dates & amounts
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if tz:
        # localize naive datetimes to timezone if requested
        df['date'] = df['date'].dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    # amount numeric
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    else:
        raise ValueError("No amount-like column found in CSV. Preprocessing requires an amount column.")
    # --- Amount parsing & sign-normalization (robust to parentheses, currency symbols, and type columns)
    # keep original for diagnostics if needed
    raw_amount = df['amount'].astype(str)

    # detect parentheses e.g. "(100.00)" -> negative
    paren_mask = raw_amount.str.contains(r'[\(\)]', regex=True)

    # strip non-numeric except minus and dot (removes currency symbols, commas, text like 'CR'/'DR')
    cleaned = raw_amount.str.replace(r'[^\d\.\-]', '', regex=True)

    # convert to numeric (coerce invalid -> NaN)
    numeric = pd.to_numeric(cleaned, errors='coerce')

    # apply parentheses sign rule
    numeric = np.where(paren_mask, -np.abs(numeric), numeric).astype(float)

    # If values are all non-negative but there is a type-like column (credit/debit),
    # attempt to infer sign from that column (common in many CSVs).
    # First, find a type-like column if present (guess common names)
    type_col = None
    for candidate in ['type', 'transaction_type', 'txn_type', 'dr_cr', 'debit_credit', 'direction']:
        if candidate in df.columns:
            type_col = candidate
            break

    if type_col is not None:
        # mark typical debit/expense indicators (case-insensitive)
        type_series = df[type_col].astype(str).str.lower()
        debit_mask = type_series.str.contains(r'debit|withdraw|expense|dr|out|debit$', na=False)

        # only apply this sign-fix if numeric values are non-negative (prevents flipping real negatives)
        if np.nanmin(numeric) >= 0:
            # flip to negative where type indicates debit/expense
            numeric = np.where(debit_mask, -np.abs(numeric), numeric)

    # final assignment back to df (keep NaNs if conversion failed)
    df['amount'] = pd.Series(numeric, index=df.index).astype(float)

    # 4) Drop rows missing critical information (date or amount)
    before_drop = len(df)
    df = df.dropna(subset=['date', 'amount']).copy()
    log['dropped_missing_date_or_amount'] = before_drop - len(df)

    # 5) Optionally drop zero-amount transactions (often empty or fees not relevant)
    if drop_zero_amounts:
        before_zero = len(df)
        df = df[df['amount'] != 0].copy()
        log['dropped_zero_amounts'] = before_zero - len(df)

    # 6) Fill text fields with placeholders
    if 'description' not in df.columns:
        df['description'] = 'unknown'
    else:
        df['description'] = df['description'].fillna('unknown').astype(str)

    if 'merchant' not in df.columns:
        df['merchant'] = 'unknown_merchant'
    else:
        df['merchant'] = df['merchant'].fillna('unknown_merchant').astype(str)

    # 7) Preserve category if present, else create column and mark missing (la nt2kad kl cat mawjuden)
    if 'category' in df.columns:
        df['category_missing'] = df['category'].isna()
    else:
        df['category'] = np.nan
        df['category_missing'] = True

    # 8) Balance handling: forward fill per account if present otherwise global ffill
    if 'balance' in df.columns:
        if 'account_id' in df.columns:
            df = df.sort_values(['account_id', 'date'])
            df['balance'] = df.groupby('account_id')['balance'].fillna(method='ffill')
        else:
            df = df.sort_values('date')
            df['balance'] = df['balance'].fillna(method='ffill')

        # 9) Clean text fields (hol 8er holek)
    df['description_clean'] = df['description'].apply(_clean_text_basic)
    df['merchant_clean'] = df['merchant'].apply(_clean_text_basic)

    # 9.5) Category normalization
    def _simple_cat_norm(c):
        if pd.isna(c) or str(c).strip() == '':
            return 'Unknown'
        return str(c).strip()

    df['category'] = df['category'].fillna('Unknown').apply(_simple_cat_norm)

    manual_map = {
        'Food': 'Food & Drink',
        'Dining': 'Food & Drink',
        'Groceries': 'Food & Drink',
        'Grocery': 'Food & Drink',
        'Restaurant': 'Food & Drink',
        'Electricity': 'Bills',
        'Water': 'Bills',
        'Internet': 'Bills',
        'Uber': 'Transport',
        'Taxi': 'Transport',
        'Bus': 'Transport',
        'Salary': 'Salary',
        'Paycheck': 'Salary',
    }

    df['category_norm'] = df['category'].map(manual_map).fillna(df['category'])



    # 10) Deduplicate exact duplicates on (date, amount, description_clean)
    before_dupes = len(df)
    df = df.drop_duplicates(subset=['date', 'amount', 'description_clean'])
    log['dropped_exact_duplicates'] = before_dupes - len(df)

    # 11) Standardize transaction type from amount 
    # sets transaction_type = 'expense' if amount < 0 else 'income'
    df['transaction_type'] = df['amount'].apply(lambda x: 'income' if x > 0 else 'expense')
    df['is_income'] = df['amount'] > 0
    # absolute and log transforms for analytics charts
    df['amount_abs'] = df['amount'].abs()
    df['amount_log1p'] = np.log1p(df['amount_abs'])

    # 12) hol for analysis 3mlton 
    df = df.sort_values('date').reset_index(drop=True)
    df['year'] = df['date'].dt.year
    # canonical month start timestamp for grouping
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['month_str'] = df['month'].dt.strftime('%Y-%m')
    df['day'] = df['date'].dt.day
    try:
        df['weekday'] = df['date'].dt.day_name()
    except Exception:
        # in some pandas builds .dt.day_name() might need locale handling
        df['weekday'] = df['date'].dt.weekday

    # 13) Simple recurring detection (heuristic)
    df['amt_rounded'] = df['amount'].round(2)
    recur_counts = df.groupby(['merchant_clean', 'amt_rounded']).size().reset_index(name='cnt')
    recurring_pairs = set(
        tuple(x) for x in recur_counts[recur_counts['cnt'] >= 3][['merchant_clean', 'amt_rounded']].values
    )
    df['recurring_candidate'] = df.apply(lambda r: (r['merchant_clean'], r['amt_rounded']) in recurring_pairs, axis=1)

    # 14) Light outlier tagging (for QA; we do not remove)
    # flag very large transactions for manual review (top 0.1% by absolute value)
    cutoff = np.percentile(df['amount_abs'].values, 99.9) if len(df) > 1000 else df['amount_abs'].quantile(0.99)
    df['large_txn_flag'] = df['amount_abs'] > cutoff
    log['large_txn_cutoff'] = float(cutoff)

    # 15) Merchant frequency (useful for reporting)
    merchant_freq = df['merchant_clean'].value_counts().to_dict()
    df['merchant_freq'] = df['merchant_clean'].map(merchant_freq).fillna(0).astype(int)

    # 16) Monthly aggregation (continuous monthly index)
    monthly = df.groupby('month').agg(
    net_amount=('amount', 'sum'),
    total_expense=('amount', lambda s: s[s < 0].abs().sum()),
    total_income=('amount', lambda s: s[s > 0].sum()),
    txn_count=('amount', 'count')
).reset_index()

    # ensure month is datetime and continuous (no gaps)
    monthly['month'] = pd.to_datetime(monthly['month'])
    if len(monthly) > 0:
        monthly = monthly.set_index('month').asfreq('MS').fillna(0).reset_index()

        # 17) Category-month aggregation
    cat_month = df.groupby(['month', 'category_norm']).agg(
        category_amount=('amount', lambda s: s[s < 0].abs().sum()),
        category_count=('amount', 'count')
    ).reset_index().rename(columns={'category_norm': 'category'})


    # 18) Save artifacts
    cleaned_path = out_path / 'transactions_clean.csv'
    monthly_path = out_path / 'monthly_agg_continuous.csv'
    cat_month_path = out_path / 'cat_month.csv'
    log_path = out_path / 'preprocessing_log.json'

    df.to_csv(cleaned_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    cat_month.to_csv(cat_month_path, index=False)

    # 19) Final log details
    log.update({
        'final_rows': len(df),
        'null_counts': df.isna().sum().to_dict(),
        'top_merchants': list(df['merchant_clean'].value_counts().head(50).index),
        'saved_paths': {
            'cleaned_csv': str(cleaned_path),
            'monthly_csv': str(monthly_path),
            'cat_month_csv': str(cat_month_path),
            'log_json': str(log_path)
        }
    })

    # write log json
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, default=str)

    if verbose:
        print(f"Preprocessing finished. Clean rows: {log['final_rows']}. Artifacts saved to: {out_path}")

    return df, monthly, cat_month, log


cleaned_df, monthly_df, cat_month_df, log = preprocess_core_transactions(
    csv_path=r"C:\Users\user\Desktop\Talktech intern\archive (4)\personal_finance_Dataset.csv",
    out_dir=r"C:\Users\user\Desktop\Talktech intern\processed",
    tz=None,
    drop_zero_amounts=True,
    verbose=True
)
print(cleaned_df.shape)
print(monthly_df.head())
print(log['final_rows'])

# 1) Make sure both income and expense exist
print(cleaned_df['transaction_type'].value_counts())

# 2) Ensure no positive expenses
print(cleaned_df.loc[cleaned_df['transaction_type'] == 'expense', 'amount'].min())

# largest (most negative) expense row
idx = cleaned_df['amount'].idxmin()
print(cleaned_df.loc[idx, ['date','amount','description','merchant_clean']])

neg_months = (monthly_df['net_amount'] < 0).sum()
total_months = len(monthly_df)
print(f"{neg_months}/{total_months} months are net negative ({neg_months/total_months:.0%})")

monthly_df.sort_values('net_amount').head(10)  # most negative first
print("rows, cols:", cleaned_df.shape)
print("nulls:\n", cleaned_df.isna().sum())

# ensure no positive values labeled as expense
bad = cleaned_df[(cleaned_df['amount'] > 0) & (cleaned_df['transaction_type'] == 'expense')]
print("Positive amount but labeled expense:", len(bad))

# ensure no negative values labeled as income
bad2 = cleaned_df[(cleaned_df['amount'] < 0) & (cleaned_df['transaction_type'] == 'income')]
print("Negative amount but labeled income:", len(bad2))

check = monthly_df['total_income'] - monthly_df['total_expense'] - monthly_df['net_amount']
print("max abs diff (should be ~0):", check.abs().max())

cat_counts = cleaned_df['category'].value_counts(dropna=False)
print(cat_counts.head(20))
print("percent missing categories:", cleaned_df['category_missing'].mean())

print("min amount (most negative):", cleaned_df['amount'].min())
print("max amount:", cleaned_df['amount'].max())

print("large txn count:", cleaned_df['large_txn_flag'].sum())
print(cleaned_df.loc[cleaned_df['large_txn_flag']].head())

print("recurring candidates (True):", cleaned_df['recurring_candidate'].sum())
cleaned_df.loc[cleaned_df['recurring_candidate']].head()


