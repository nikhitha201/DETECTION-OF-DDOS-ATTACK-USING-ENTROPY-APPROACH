from sklearn.cluster import KMeans

# Very small dataset
data = [[1, 2], [1, 4], [5, 8], [8, 8]]

# Create KMeans with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)

# Print cluster centers and labels
print("Cluster Centers:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
