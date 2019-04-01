import numpy as np 
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

def dti_cluster(mat,n_clusters=3):
    scaled = scale(mat.toarray())
    CC = np.nan_to_num(np.corrcoef(scaled))
    return KMeans(n_clusters=n_clusters).fit_predict(CC) + 1

