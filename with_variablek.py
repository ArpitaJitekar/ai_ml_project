import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==========================================================
# STEP 1: Load & Prepare Data
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Arpita\Downloads\archive (3)\online_retail_II.csv", encoding='latin1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df.dropna(subset=['InvoiceDate', 'Customer ID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['Price']

    snapshot_date = df['InvoiceDate'].max()
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

rfm = load_data()

# ==========================================================
# STEP 2: Preprocessing (log + scale)
# ==========================================================
rfm['Monetary'] = np.log1p(rfm['Monetary'])
rfm['Frequency'] = np.log1p(rfm['Frequency'])

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# ==========================================================
# STEP 3: Weighted K-Means Implementation
# ==========================================================
def weighted_kmeans(X, k=5, weights=None, max_iters=100):
    n_samples, n_features = X.shape
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(n_samples, size=k, replace=False)]

    if weights is None:
        weights = np.ones(n_features)
    weights = np.array(weights)

    for _ in range(max_iters):
        # Weighted distance
        dists = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2 * weights).sum(axis=2))
        labels = np.argmin(dists, axis=1)

        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return centroids, labels

# ==========================================================
# STEP 4: Find Best k (Dynamic Selection)
# ==========================================================
def find_best_k(X, weights, k_min=2, k_max=10):
    distortions, sil_scores = [], []
    for k in range(k_min, k_max + 1):
        centroids, labels = weighted_kmeans(X, k=k, weights=weights)
        dist = np.mean(np.min(np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2 * weights).sum(axis=2)), axis=1))
        distortions.append(dist)
        sil_scores.append(silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1)

    k_range = np.arange(k_min, k_max + 1)
    sil_scores = np.array(sil_scores)
    distortions = np.array(distortions)

    # Silhouette candidate
    sil_candidate = k_range[np.argmax(sil_scores)]
    if sil_candidate <= 2:
        sil_candidate = 3

    # Elbow candidate (largest drop)
    diffs = np.diff(distortions)
    elbow_candidate = k_range[np.argmin(diffs) + 1]

    # Combine both
    if abs(elbow_candidate - sil_candidate) <= 1:
        best_k = elbow_candidate
    else:
        best_k = int(np.median([elbow_candidate, sil_candidate, 4]))  # fallback mid value

    best_k = max(3, best_k)
    return best_k, distortions, sil_scores

# ==========================================================
# STEP 5: Train Final Model
# ==========================================================
weights = [1.5, 1.0, 2.0]  # Recency, Frequency, Monetary
best_k, distortions, sil_scores = find_best_k(rfm_scaled, weights)
centroids, labels = weighted_kmeans(rfm_scaled, k=best_k, weights=weights)
rfm['Cluster'] = labels

# ==========================================================
# STEP 6: Dynamic Cluster Naming
# ==========================================================
cluster_summary = rfm.groupby('Cluster').mean().reset_index()

# Normalize each feature for comparison
rfm_norm = (cluster_summary[['Recency', 'Frequency', 'Monetary']] - cluster_summary[['Recency', 'Frequency', 'Monetary']].min()) / \
           (cluster_summary[['Recency', 'Frequency', 'Monetary']].max() - cluster_summary[['Recency', 'Frequency', 'Monetary']].min())

names = []
for i, row in rfm_norm.iterrows():
    r, f, m = row['Recency'], row['Frequency'], row['Monetary']

    if r < 0.3 and f > 0.7 and m > 0.7:
        names.append(" Loyal Big Spenders")
    elif r < 0.4 and m > 0.7 and f < 0.4:
        names.append(" High-Value but Infrequent")
    elif r < 0.6 and f > 0.5:
        names.append(" Regular Buyers")
    elif r > 0.7 and m < 0.4 and f < 0.4:
        names.append("Inactive Customers")
    else:
        names.append(" Lost Customers")

cluster_summary['Segment'] = names
cluster_map = dict(zip(cluster_summary['Cluster'], cluster_summary['Segment']))
rfm['Segment'] = rfm['Cluster'].map(cluster_map)

print(f" Best number of clusters (k): {best_k}")
print("\n Cluster Summary:")
print(cluster_summary[['Cluster', 'Recency', 'Frequency', 'Monetary', 'Segment']])

# ==========================================================
# STEP 7: Visualizations (Separate, not in Streamlit)
# ==========================================================
def plot_elbow_silhouette(distortions, sil_scores, k_min=2):
    ks = range(k_min, k_min + len(distortions))
    fig, ax1 = plt.subplots()
    ax1.plot(ks, distortions, 'bo-', label='Elbow (Distortion)')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Distortion', color='b')
    ax2 = ax1.twinx()
    ax2.plot(ks, sil_scores, 'ro-', label='Silhouette')
    ax2.set_ylabel('Silhouette', color='r')
    plt.title("Elbow & Silhouette Method")
    plt.show()

def plot_clusters(X, labels, centroids):
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel("Frequency (scaled)")
    plt.ylabel("Monetary (scaled)")
    plt.title("Customer Clusters with Centroids")
    plt.legend()
    plt.show()

# --- Show plots ---
plot_elbow_silhouette(distortions, sil_scores)
plot_clusters(rfm_scaled, labels, centroids)

# ==========================================================
# STEP 8: Streamlit Minimal App
# ==========================================================
st.title("Customer Segmentation (Weighted K-Means)")
st.markdown("Enter customer RFM values below to predict their segment:")

recency = st.number_input("Recency (days since last purchase):", min_value=0.0)
frequency = st.number_input("Frequency (no. of purchases):", min_value=0.0)
monetary = st.number_input("Monetary (total spending):", min_value=0.0)

if st.button("Predict Segment"):
    input_data = np.array([[recency, frequency, monetary]])
    input_data[:, 1:] = np.log1p(input_data[:, 1:])  # Log-transform for scale matching
    input_scaled = scaler.transform(input_data)
    dists = np.sqrt(((input_scaled - centroids) ** 2 * weights).sum(axis=1))
    cluster_idx = np.argmin(dists)
    predicted_segment = cluster_map.get(cluster_idx, "Unknown")
    st.success(f"Predicted Customer Type: **{predicted_segment}**")

st.info("Model dynamically determines the optimal number of clusters using Elbow + Silhouette.")
