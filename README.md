# Customer Segmentation with Weighted-K-Means  
*A practical project for AI/ML coursework*

## 🎯 Project Overview  
This project uses the “Online Retail II” dataset to perform customer segmentation via RFM (Recency, Frequency, Monetary) analysis and a custom implementation of Weighted K-Means clustering. The goal is to classify customers into meaningful segments (e.g., “Loyal Big Spenders”, “Infrequent but High Value”, etc.) and build a small interactive demo for real-time predictions.  

## 📊 Why this matters  
Proper customer segmentation helps businesses target marketing, improve retention, and allocate resources more effectively. By adding weights to features in K-Means, we emphasise features that matter most (e.g., monetary value vs. recency) and achieve more meaningful clusters.  

## 🧰 Dataset  
- Source: [UCI Online Retail II dataset] – Sheet “Year 2009-2010”.  
- Columns used: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.  
- Preprocessing steps: filter UK customers only, drop canceled orders, negative quantities, handle missing Customer IDs, create RFM features.  

## 🔧 Methodology  
### 1. RFM Feature Engineering  
- **Recency**: Days since last purchase per customer.  
- **Frequency**: Number of unique invoices per customer.  
- **Monetary**: Total spending per customer.  

### 2. Weighted K-Means Clustering  
- We define weights \(w_f\) for each feature \(f\) so that:  
  \[
     \text{Distance}(x_i, \mu_j) = \sqrt{ \sum_{f=1}^n w_f \cdot (x_{if} - \mu_{jf})^2 }
  \]  
- Centroid update is standard:  
  \[
     \mu_j = \frac{1}{\lvert C_j \rvert} \sum_{x_i \in C_j} x_i
  \]  
- We select the optimal number of clusters \(k\) by computing both the Elbow method and the Silhouette score.  

## 📈 Visualisations  
- Histograms and boxplots for RFM distributions (original vs log-transformed).  
- Plot of Elbow curve + Silhouette scores vs k.  
- Final cluster scatter plot with centroids labelled.  
- Interactive Streamlit app (see next section).  

## 🧩 Segment Labelling  
After clustering, the algorithm assigns interpretive labels based on normalized feature values:  
- **Loyal Big Spenders**: High monetary, high frequency, low recency.  
- **High-Value but Infrequent**: High monetary, low frequency.  
- **Regular Buyers**: Moderate across features.  
- **Inactive Customers**: Long recency, low frequency.  
- **Lost Customers**: Very long recency, minimal activity.  

## 🚀 Interactive Demo (Streamlit)  
A small web app lets you input Recency, Frequency, Monetary values and immediately see the predicted customer segment based on the trained centroids.  
**How to launch:**  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit openpyxl  
streamlit run app.py
