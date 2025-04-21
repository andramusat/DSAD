import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rawRata = pd.read_csv('D:\subiecte_examen\subiect5\dateIN\Rata.csv', index_col=0)
rawCoduri = pd.read_csv('D:\subiecte_examen\subiect5\dateIN\CoduriTariExtins.csv', index_col=0)
labels = list(rawRata.columns[1:].values)

rawCoduri.rename(columns={'Country_Letter_code': 'Three_Letter_Country_Code', 'Country': 'Country_Name'})

merged = rawRata.merge(rawCoduri, left_index=True, right_index=True)[['Country_Name', 'Continent'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
print(merged)

# A1

cerinta1 = merged[merged['RS'] < np.average(merged['RS'])][['Country_Name', 'RS']].sort_values('RS', ascending=False)
cerinta1.to_csv('D:\subiecte_examen\subiect5\dateOUT\Cerinta1.csv', index_label='Three_Letter_Country_Code')

# A2

cerinta2 = merged.groupby('Continent') \
    .apply(lambda df: pd.Series({rata: df[rata].idxmax() for rata in labels}), include_groups=False)
cerinta2.to_csv('D:\subiecte_examen\subiect5\dateOUT\Cerinta2.csv', index_label='Continent_Name')

# B1

x = StandardScaler().fit_transform(merged[labels])

pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
pve = pca.explained_variance_ratio_

var_cum = np.cumsum(alpha)
pve_cum = np.cumsum(pve)

pd.DataFrame(data={'Varianta componentelor': alpha,
                   'Varianta cumulata': var_cum,
                   'Procentul de varianta explicata': pve,
                   'Procentul cumulat': pve_cum}) \
    .to_csv('D:\subiecte_examen\subiect5\dateOUT\Varianta.csv', index=False)

# B2

plt.figure(figsize=(10, 10))
plt.title('Varianta explicata de catre componente')
labels = ['C' + str(i + 1) for i in range(len(alpha))]
plt.plot(labels, alpha, 'bo-')
plt.axhline(1, c='r')
plt.show()

# B3

a = pca.components_.T
Rxc = a * np.sqrt(alpha)
communalities = np.cumsum(Rxc * Rxc, axis=1)
communalities_df = pd.DataFrame(data=communalities, index=labels,
                                columns=['C' + str(i + 1) for i in range(communalities.shape[1])])

plt.figure(figsize=(10, 10))
plt.title('Corelograma corelatilor')
sb.heatmap(communalities_df, vmin=-1, vmax=1, annot=True, cmap='bwr')
plt.show()
