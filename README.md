# ai_ml_project

# 🛍️ Customer Segmentation with Weighted K-Means

This project performs customer segmentation using RFM (Recency, Frequency, Monetary) analysis and a custom Weighted K-Means clustering algorithm. It includes data preprocessing, dynamic cluster selection, interactive visualizations, and a Streamlit app for real-time segment prediction.

---

## 📁 Dataset

- **Source**: [UCI Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- **File Used**: `online_retail_II.xlsx` (Sheet: `Year 2009-2010`)
- **Fields**: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country

---

## 📊 RFM Analysis

RFM metrics are calculated per customer:
- **Recency**: Days since last purchase
- **Frequency**: Number of unique invoices
- **Monetary**: Total spending

---

## ⚙️ Weighted K-Means Clustering

### ✅ Features:
- Custom implementation of K-Means with feature-level weights
- Weighted Euclidean distance:
  \[
  \text{Distance}(x_i, \mu_j) = \sqrt{ \sum_{f=1}^{n} w_f \cdot (x_{if} - \mu_{jf})^2 }
  \]
- Centroid update:
  \[
  \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
  \]
- Dynamic selection of optimal `k` using Elbow + Silhouette methods

---

## 📈 Visualizations

- Distribution plots (histograms, boxplots)
- Log-transformed feature plots
- Elbow & Silhouette score comparison
- Cluster scatter plot with centroids

---

## 🧠 Segment Naming

Clusters are labeled based on normalized RFM values:
- Loyal Big Spenders
- High-Value but Infrequent
- Regular Buyers
- Inactive Customers
- Lost Customers

---

## 🌐 Streamlit App

### Features:
- Input Recency, Frequency, Monetary values
- Predict customer segment in real-time
- Displays predicted segment using trained centroids

### Run the app:
```bash
streamlit run app.py
