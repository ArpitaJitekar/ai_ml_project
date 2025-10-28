# --- Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#pip install openpyxl

# --- Load dataset ---
file_path = r"C:\Users\DEEPIKA\Downloads\archive (3)\online_retail_II.xlsx"
online_retail = pd.read_excel(file_path, sheet_name='Year 2009-2010')

# --- Step 0: Clean data ---
# Drop rows without Customer ID
online_retail = online_retail.dropna(subset=['Customer ID'])

# Ensure date column is datetime
online_retail['InvoiceDate'] = pd.to_datetime(online_retail['InvoiceDate'])

# Create TotalPrice column
online_retail['TotalPrice'] = online_retail['Quantity'] * online_retail['Price']

# --- Step 1: Create RFM table ---
latest_date = online_retail['InvoiceDate'].max()

rfm = online_retail.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
    'Invoice': 'nunique',                                  # Frequency
    'TotalPrice': 'sum'                                    # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# --- Step 2: Summary statistics ---
print("📊 Summary Statistics:\n", rfm[['Recency', 'Frequency', 'Monetary']].describe())

# --- Step 3: Skewness ---
print("\n📈 Skewness of each feature:")
print(rfm[['Recency', 'Frequency', 'Monetary']].skew())

# --- Step 4: Visualize distributions ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
cols = ['Recency', 'Frequency', 'Monetary']

# Histograms
for i, col in enumerate(cols):
    sns.histplot(rfm[col], bins=50, kde=True, ax=axes[0, i], color='lightblue')
    axes[0, i].set_title(f"{col} Distribution")

# Boxplots
for i, col in enumerate(cols):
    sns.boxplot(x=rfm[col], ax=axes[1, i], color='lavender')
    axes[1, i].set_title(f"{col} Boxplot")

plt.tight_layout()
plt.show()

# --- Step 5: Check effect of log-transform ---
rfm_log = rfm.copy()
for col in ['Frequency', 'Monetary']:
    rfm_log[col] = np.log1p(rfm[col])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(rfm_log['Frequency'], bins=40, kde=True, ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Frequency after log-transform')
sns.histplot(rfm_log['Monetary'], bins=40, kde=True, ax=axes[1], color='salmon', edgecolor='black')
axes[1].set_title('Monetary after log-transform')
plt.show()
