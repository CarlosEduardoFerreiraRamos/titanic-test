"""
Cluster is similar to classification.

Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you
are looking for, and you are trying to identify some segments or clusters in your data. When you use
clustering algorithms on your dataset, unexpected things can suddenly pop up like structures,
clusters and groupings you would have never thought of otherwise.
"""

"""
K-Means
steps:

1 - choose the number of clusters.

2 - select at random k points, the centroids (not necessarily from your dataset)

3 - assing rach data point to the closest centroid. This processes forms K clusters.

4 - compute and palce the new centroid of each cluster.

5 - reassign each data point ti the bew closest centroid.
	# If any reassignment took place, got o step 4, otherwise FIN
"""

"""
Random Initializatio trap

the solution to this is the K-Means++
"""

"""
Chooseing the Right number of Clusters

WCSS

used in the elbow method
"""
#  max_iter = 300, n_init = 10 props with the default values
from sklean.cluster import KMeans;

wcssList = [];

for item in range(1,11):
	kmeans = KMeans(n_clusters = item, ini = 'k-means++', max_iter = 300, n_init = 10, random_state = 0);
	jmeans.fit(x)
	wcss.append(kmeans.inertia_)

# aplying k means

kmeans = KMeans(n_clusters = determinated_number, ini = 'k-means++', max_iter = 300, n_init = 10, random_state = 0);
y_kmeans = kmeans.fit_predicti(x)



"""
HC clustering

1- Aglomerative: Build up

1- Divisive: divide down


steps:

1 - make a data point a single-point cluster. Forming N clusters;

2 - Take the two closest data points and make them on cluster. forming N -1;

3 - Take the two closest clusters and make them on cluster. forming N -1;

4 - repeat step 3 until there is only one cluster.
	# then FIN

"""

"""
Classifers pros and cons        PROS  											CONS

K-Means:    			  Simple to undertand, easily adaptable,            Need to choose the number of clusters
						  works well on large or small datasets,
						  fast, efficient and performatic  			
						  													  
Hierarchical Clustering:  the optimal number of clusters can be 			Not appropriate for large Datasets
						  obtained by the model  it self, pratical  			
						  visualization with the dendogram
"""
import scipy.cluster as sch;

from sklearn.cluster import AgglomerativeClustering; 

class HCluster(object):

	def __init__(self):
		self.cluster = None;
		self.acluster = None;

	def create_hcluster(self, x_train):
		self.cluster = sch.dendrogram(sch.linkage(x_train, method = 'ward'));

	def create_ahcluster(self, n_clusters):
		self.acluster = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'ward');

	def predict_ahcluster(self, x_test):
		return self.acluster.fit_predict(x_test);