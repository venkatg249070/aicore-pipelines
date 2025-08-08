import os
#
# Variables
DATA_PATH = '/app/data/FMCG_2022_2024.csv'
DT_MAX_DEPTH = int(os.getenv('DT_MAX_DEPTH'))
MODEL_PATH = '/app/model/model.pkl'
#
# Load Datasets
import pandas as pd
df = pd.read_csv(DATA_PATH)

# Data Preprocessing (incorporating steps from previous cells)
categorical_cols = ['sku', 'brand', 'segment', 'category', 'channel', 'region', 'pack_type']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

numerical_cols = ['price_unit', 'delivery_days', 'stock_available', 'delivered_qty', 'units_sold']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Create 'Score' column (as done previously)
df_encoded['Score'] = df_encoded['delivered_qty'] * df_encoded['units_sold']

# Now drop 'Score' and 'date' as features and assign 'Score' to y
X = df_encoded.drop(['Score', 'date'], axis=1)
y = df_encoded['Score']
#
# Partition into Train and test dataset
from sklearn.model_selection import train_test_split
# Using a test size of 0.2 to be consistent with previous splits
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
#
# Init model
from sklearn.tree import DecisionTreeRegressor
# Ensure StandardScaler is imported for preprocessing steps
from sklearn.preprocessing import StandardScaler
clf = DecisionTreeRegressor(max_depth=DT_MAX_DEPTH, random_state=42) # Added random_state for reproducibility
#
# Train model
clf.fit(train_x, train_y)
#
# Test model
test_r2_score = clf.score(test_x, test_y)
# Output will be available in logs of SAP AI Core.
# Not the ideal way of storing /reporting metrics in SAP AI Core, but that is not the focus this tutorial
print(f"Test Data Score {test_r2_score}")
#
# Save model
import pickle
pickle.dump(clf, open(MODEL_PATH, 'wb'))
