# Full preprocessing for synthetic AML transactions - Phase 8
# Python 3.13 / Pandas 4 / scikit-learn 1.2+ compatible

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1️⃣ Load dataset
df = pd.read_csv("synthetic_aml_transactions.csv")

# 2️⃣ Remove duplicates
df = df.drop_duplicates()

# 3️⃣ Handle missing values
# Numerical columns → fill with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object', 'string']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# 4️⃣ Preprocessing pipeline
# Numerical → StandardScaler
# Categorical → OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)

# Apply preprocessing
df_processed = preprocessor.fit_transform(df)

# 5️⃣ Optional: Convert processed data to DataFrame with column names
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = list(num_cols) + list(cat_feature_names)
df_processed = pd.DataFrame(df_processed, columns=all_feature_names)

# 6️⃣ Save preprocessed data to CSV
df_processed.to_csv("synthetic_aml_transactions_preprocessed.csv", index=False)

print("✅ Preprocessing complete!")
print("Processed data shape:", df_processed.shape)




