#  Customer Segmentation Using Weighted K-Means

---

## Overview

This project performs **Customer Segmentation** on the **Online Retail II (UCI)** dataset using **Weighted K-Means Clustering** based on **RFM (Recency, Frequency, Monetary)** analysis.
It groups customers into meaningful behavioral segments, providing insight into purchase patterns for targeted marketing and retention strategies.

---

##  Objectives

* To preprocess and analyze retail data using RFM features.
* To apply **Weighted K-Means** to improve clustering accuracy.
* To determine the optimal number of clusters using **Elbow** and **Silhouette** methods.
* To build a **Streamlit web app** for real-time customer segment prediction.

---

##  Dataset Information

* **Source:** [UCI Machine Learning Repository – Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail+ii)
* **Attributes Used:**

  * InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

###  Preprocessing Steps

1. Filter data for **United Kingdom** customers.
2. Remove **cancelled orders** and **negative quantities**.
3. Drop records with **missing Customer IDs**.
4. Compute **RFM features**:

   * **Recency** – Days since the last purchase
   * **Frequency** – Number of unique invoices
   * **Monetary** – Total amount spent

---

##  Methodology

###  RFM Feature Computation

Each customer’s RFM score is calculated as follows:

[
\text{Recency} = (\text{Max InvoiceDate}) - (\text{Last Purchase Date})
]

[
\text{Frequency} = \text{Count of Unique Invoices}
]

[
\text{Monetary} = \sum (\text{Quantity} \times \text{UnitPrice})
]

---

### 2️ Weighted K-Means Clustering

Unlike standard K-Means, **Weighted K-Means** assigns a weight to each feature to control its influence on the distance metric.

#### Distance Calculation

[
D(x_i, \mu_j) = \sqrt{ \sum_{f=1}^{n} w_f \cdot (x_{if} - \mu_{jf})^2 }
]

where:

* ( w_f ) = weight for feature ( f )
* ( x_i ) = data point
* ( \mu_j ) = cluster centroid

#### Centroid Update

[
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
]

---

### 3️ Optimal k Selection

* **Elbow Method:**
  Plot the **Within-Cluster Sum of Squares (WCSS)** versus k and choose the point where the curve “bends”.

* **Silhouette Score:**
  Measure how similar each sample is to its own cluster compared to others.

[
S = \frac{b - a}{\max(a, b)}
]

where:

* ( a ) = mean intra-cluster distance
* ( b ) = mean nearest-cluster distance

---

##  Segment Interpretation

| Cluster | Customer Type             | Characteristics                            |
| ------- | ------------------------- | ------------------------------------------ |
| 1       | **Loyal Big Spenders**    | High Monetary, High Frequency, Low Recency |
| 2       | **High-Value Occasional** | High Monetary, Low Frequency               |
| 3       | **Regular Buyers**        | Moderate across all features               |
| 4       | **Inactive Customers**    | Low Frequency, Long Recency                |
| 5       | **Lost Customers**        | No recent purchases, minimal activity      |

---

##  Visualizations

* **RFM Distributions:** Histograms before & after log transformation.
* **Outlier Detection:** Boxplots to visualize anomalies.
* **Elbow Curve & Silhouette Plot:** To identify optimal k.
* **Cluster Scatterplot:** Clustered customers with labeled centroids.

---

##  Streamlit App Demo

Run an interactive web app to input R, F, M values and get real-time predictions.

###  Launch Steps

1. **Install required packages:**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit openpyxl
   ```
2. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```
3. **Access in browser:**
   Visit [http://localhost:8501](http://localhost:8501)

---

##  How to Execute the Project

###  Step-by-Step Execution

1. **Clone the repository**

   ```bash
   git clone https://github.com/ArpitaJitekar/ai_ml_project.git
   cd ai_ml_project
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *(If no requirements file, use the command under “Launch Steps”)*
3. **Prepare the dataset**

   * Place `online_retail_II.xlsx` in the project root (sheet 2009–2010).
4. **Run the analysis script**

   ```bash
   python with_variablek.py
   ```

   * Outputs plots, cluster labels, and performance metrics.
5. **(Optional) Run Streamlit interface**

   ```bash
   streamlit run app.py
   ```
6. **View results**

   * Clustering results appear in console and plots folder.
   * Web interface shows predicted customer segment.

---

## 🧪 Results & Inferences

* Weighted K-Means improves interpretability by emphasizing **Monetary Value**.
* Optimal clusters (k) typically found between 4 and 6.
* Segments align well with business intuition (Loyal vs Lost Customers).
* Log transformation stabilizes RFM distributions and reduces skewness.

---

##  Repository Structure

```
ai_ml_project/
│
├── app.py                    # Streamlit interface
├── with_variablek.py         # Weighted K-Means clustering script
├── skewness_and_outlier/     # Outlier & distribution analysis
├── online_retail_II.xlsx     # Dataset (sheet 2009-2010)
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

---

##  Future Improvements

* Extend to other countries for cross-regional analysis.
* Integrate **DBSCAN** or **Gaussian Mixture Models** for comparison.
* Add dashboard visuals to Streamlit app (cluster size, average RFM).
* Deploy app on **Streamlit Cloud** for public access.

---




