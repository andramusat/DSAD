import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


rawAlcohol = pd.read_csv(filepath_or_buffer='D:\\subiecte_examen\\subiect4\\dateIN\\alcohol.csv', index_col=0)
rawCoduri = pd.read_csv(filepath_or_buffer='D:\subiecte_examen\subiect4\dateIN\CoduriTariExtins.csv', index_col=0)
labels = list(rawAlcohol.columns[1:].values)

merged = rawAlcohol.merge(rawCoduri, left_index=True, right_index=True)[['Code', 'Continent']+labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
print(merged)

# A1
cerinta1 = merged.apply(lambda row: np.average(row[labels]), axis=1).sort_values(ascending=False)
cerinta1_df = pd.DataFrame({
    'Country': cerinta1.index,
    'Code': merged.loc[cerinta1.index, 'Code'],
    'Consum Mediu': cerinta1.values
})
cerinta1_df.to_csv('D:\subiecte_examen\subiect4\dateOUT\cerinta1.csv', index=False)

# A2

cerinta2 = merged[['Continent']+labels].groupby('Continent').mean().idxmax(axis=1)
cerinta2_df = pd.DataFrame({'Continent_Name':cerinta2.index, 'Anul':cerinta2.values})
cerinta2_df.to_csv('D:\subiecte_examen\subiect4\dateOUT\cerinta2.csv', index=False)

# B1

x = StandardScaler().fit_transform(merged[labels])

HC = linkage(x, method='ward')
print(HC)

# B2
cat = fcluster(HC, 5, criterion='maxclust')
clusters = ['C' + str(i) for i in cat]

merged['Clusters'] = clusters
merged[['Code', 'Clusters']].to_csv('D:\subiecte_examen\subiect4\dateOUT\p4.csv', index_label='Country')

# B3

pca = PCA()
C = pca.fit_transform(x)

kmeans = KMeans(n_clusters=5, n_init=10)
kmean_labels = kmeans.fit_predict(C)
plt.figure(figsize=(8,6))
plt.scatter(C[:, 0], C[:, 1], c=kmean_labels, cmap='viridis')
plt.xlabel("Componenta Principală 1")  
plt.ylabel("Componenta Principală 2")
plt.title("K-means Clustering on PCA Data")
plt.show()
