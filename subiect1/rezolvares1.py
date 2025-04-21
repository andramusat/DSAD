import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

rawAlcohol = pd.read_csv(filepath_or_buffer='subiect1/dateIN/alcohol.csv', index_col=0)
rawCoduri = pd.read_csv(filepath_or_buffer='subiect1/dateIN/CoduriTariExtins.csv', index_col=0)

labels = list(rawAlcohol.columns[1:].values)

rawAlcohol.rename(columns={'Entity':'Tari'})
merged = rawAlcohol.merge(rawCoduri, right_index=True, left_index=True)[['Code', 'Continent']+labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
print(merged)

# A1

cerinta1 = merged.apply(lambda row: np.average(row[labels]), axis=1)
cerinta1_df = pd.DataFrame({'Code':cerinta1.index, 'Media':cerinta1.values})
cerinta1_df.to_csv('subiect1/dateOUT/Cerinta1.csv', index=False)

# A2

cerinta2 = merged[['Continent']+labels].groupby('Continent').mean().idxmax(axis=1)
cerinta2_df = pd.DataFrame({'Continent':cerinta2.index, 'Anul':cerinta2.values})
cerinta2_df.to_csv('subiect1/dateOUT/Cerinta2.csv', index=False)

# B1

x = StandardScaler().fit_transform(merged[labels])

HC = linkage(x, method='ward')
print(HC)

n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n-1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j,2] + HC[j+1,2]) / 2
print(t)
# B2
plt.figure(figsize=(12,12))
plt.title('Dendogram')
dendrogram(HC, leaf_rotation=30, labels=merged.index.values)
plt.axhline(y=t, color='r', linestyle='--', linewidth=2)
plt.show()

# B3

cat = fcluster(HC,  n - j, criterion='maxclust')
clusters = ['C' + str(i) for i in cat]

merged['Clusters'] = clusters
merged['Clusters'].to_csv('subiect1/dateOUT/popt.csv', index=False)