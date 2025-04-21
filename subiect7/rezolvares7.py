import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rawIndicators = pd.read_csv('D:\subiecte_examen\subiect7\dateIN\GlobalIndicatorsPerCapita_2021.csv', index_col=0)
rawContinents = pd.read_csv('D:\subiecte_examen\subiect7\dateIN\CountryContinents (1).csv', index_col=0)
labels = list(rawIndicators.columns[1:].values)
indexes = list(rawIndicators.index.values)

rawContinents.rename(columns={'CountryID':'CountryId'})

merged = rawIndicators.merge(rawContinents, left_index=True, right_index=True) \
.drop('Country_y', axis=1).rename(columns={'Country_x': 'Country'})[['Continent', 'Country'] + labels]
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
print(merged)

# A1

labValAdaugata = list(merged.columns.values[-7:])
cerinta1 = merged[['Country'] + labValAdaugata].set_index('Country', append=True).sum(axis=1)
cerinta1.to_csv('D:\subiecte_examen\subiect7\dateOUT\Cerinta1.csv', index_label=['CountryID', 'Country', 'Valoare Adaugata'])

# A2

merged[['Continent'] + labels] \
    .groupby('Continent') \
    .apply(func=lambda df: pd.Series({ind: np.round(np.std(df[ind]) / np.mean(df[ind]) * 100, 2) for ind in labels}), include_groups=False) \
    .to_csv('D:\subiecte_examen\subiect7\dateOUT\Cerinta2.csv')

# B1

x = StandardScaler().fit_transform(merged[labels])

pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
print(alpha)

# B2

scores = C / np.sqrt(alpha)
pd.DataFrame(data=np.round(scores, 2), index=indexes, columns=labels).to_csv('D:\subiecte_examen\subiect7\dateOUT\scoruri.csv')

# B3

plt.figure(figsize=(12, 9))
plt.title('Scoruri')
plt.scatter(scores[:, 0], scores[:, 1])
plt.show()

# C

factorLoadings = pd.read_csv('D:\subiecte_examen\subiect7\dateIN\g20.csv', index_col=0)
communalities = np.cumsum(factorLoadings * factorLoadings, axis=1)
print(communalities.sum().idxmax())

