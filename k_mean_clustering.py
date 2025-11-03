# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 2: Load the Dataset
# Make sure 'Mall_Customers.csv' is in the same folder
dataset = pd.read_csv('Mall_Customers.csv')

# Step 3: Display Basic Info
print("========== Dataset Information ==========")
print(dataset.info())
print("\n========== First 5 Rows ==========")
print(dataset.head())

# Step 4: Select Relevant Features
# We'll use Annual Income (k$) and Spending Score (1–100)
X = dataset.iloc[:, [3, 4]].values

# Step 5: Feature Scaling (Optional but helps with clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Using the Elbow Method to find the Optimal Number of Clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Step 7: Plot the Elbow Graph
plt.figure(figsize=(6,4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='blue')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# Step 8: Apply K-Means with Optimal Clusters (k = 5)
kmeans = KMeans(n_clusters=5, init='k-means++',
                max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 9: Add Cluster Labels to Dataset
dataset['Cluster'] = y_kmeans

# Step 10: Visualize the Clusters
plt.figure(figsize=(6,5))

# Cluster 1
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1],
            s=100, c='red', label='Cluster 1')

# Cluster 2
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')

# Cluster 3
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')

# Cluster 4
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1],
            s=100, c='cyan', label='Cluster 4')

# Cluster 5
plt.scatter(X_scaled[y_kmeans == 4, 0], X_scaled[y_kmeans == 4, 1],
            s=100, c='magenta', label='Cluster 5')

# Step 11: Plot Cluster Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', marker='*', label='Centroids')

plt.title('Clusters of Mall Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# Step 12: Display Final Clustered Data
print("\n========== Clustered Data Sample ==========")
print(dataset.head())