import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler

rawAir = pd.read_csv('D:\subiecte_examen\subiect6\dateIN\AirQuality.csv', index_col=0)
rawContinent = pd.read_csv('D:\subiecte_examen\subiect6\dateIN\CountryContinents.csv', index_col=0)
labels = list(rawAir.columns[1:].values)
rawContinent.rename(columns={'CountryID':'CountryId'})

merged = rawAir.merge(rawContinent, left_index=True, right_index=True).drop('Country_y', axis=1).rename(columns={'Country_x':'Country'})[['Country', 'Continent']+labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
print(merged)

# A1

cerinta1 = merged[['Country'] + labels].set_index('Country').idxmax(axis=0)
cerinta1_df = pd.DataFrame({'Indicator':cerinta1.index, 'Country':cerinta1.values})
cerinta1_df.to_csv('D:\subiecte_examen\subiect6\dateOUT\Cerinta1.csv', index=False)

# A2

cerinta2 = merged.set_index('Country').groupby(['Continent'])\
    .apply(func=lambda df: pd.Series({ind: df[ind].idxmax() for ind in labels}), include_groups=False)
cerinta2.to_csv('D:\subiecte_examen\subiect6\dateOUT\Cerinta2.csv', index_label='Continent')

# B1

x = StandardScaler().fit_transform(rawAir[labels])

HC = linkage(x, method='ward')
print(HC)

# B2

n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n - 1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j, 2] + HC[j + 1, 2]) / 2

plt.figure(figsize=(12, 7))
plt.title('Dendogram')
dendrogram(HC, leaf_rotation=30, labels=merged['Country'].values)
plt.axhline(t, c='r')
plt.show()

# B3

cat = fcluster(HC, n - j, criterion='maxclust')
clusters = ['C' + str(i) for i in cat]

merged['Cluster'] = clusters
merged[['Country', 'Cluster']].to_csv('D:\subiecte_examen\subiect6\dateOUT\popt.csv')
