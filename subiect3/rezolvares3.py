import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rawMortalitate = pd.read_csv(filepath_or_buffer='D:\subiecte_examen\subiect3\dateIN\Mortalitate.csv', index_col=0)
rawCoduri = pd.read_csv(filepath_or_buffer='D:\subiecte_examen\subiect3\dateIN\CoduriTariExtins.csv', index_col=0)
labels = list(rawMortalitate.columns.values)

mortalitate = rawMortalitate.rename(columns={'Tara':'Tari'})
merged = mortalitate.merge(rawCoduri, right_index=True, left_index=True)[['Continent']+labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
print(merged)

# A1
cerinta1 = merged[merged['RS']<0]
cerinta1[['RS']].to_csv('D:\subiecte_examen\subiect3\dateOUT\cerinta1.csv', index_label='Tari')

# A2

cerinta2 = merged.groupby('Continent').mean()
cerinta2.to_csv('D:\subiecte_examen\subiect3\dateOUT\cerinta2.csv', index_label='Continent')

# B1

x = StandardScaler().fit_transform(merged[labels])

pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_

print(alpha)

# B2

scores = C / np.sqrt(alpha)
scores_df = pd.DataFrame(data=scores, index=mortalitate.index.values, columns=['C' + str(i+1) for i in range(C.shape[1])])
scores_df.to_csv('D:\subiecte_examen\subiect3\dateOUT\scoruri.csv', index=False)

# B3

plt.figure(figsize=(12,12))
plt.title('Scoruri')
plt.scatter(scores[:,0], scores[:,1])
plt.show()