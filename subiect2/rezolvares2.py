import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

ind = pd.read_csv(filepath_or_buffer='D:\subiecte_examen\subiect2\dateIN\Industrie.csv', index_col=0)
pop = pd.read_csv(filepath_or_buffer='D:\subiecte_examen\subiect2\dateIN\PopulatieLocalitati.csv', index_col=0)
labels = list(ind.columns[1:].values)

merged = ind.merge(pop, right_index=True, left_index=True).\
    rename(columns={'Localitate_x':'Localitate'})[['Judet','Localitate','Populatie']+labels]
print(merged)
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)

# A1

cerinta1 = merged[['Localitate', 'Populatie']+labels].apply(lambda row: row[labels] / row['Populatie'], axis=1)
cerinta1.to_csv('D:\subiecte_examen\subiect2\dateOUT\Cerinta1.csv', index_label=['Siruta', 'Localitate'])

# A2

cerinta2 = merged[['Judet']+labels].groupby('Judet').sum()
cerinta2['Cifra de afaceri'] = cerinta2.max(axis=1)
cerinta2['Activitate'] = cerinta2.idxmax(axis=1)
cerinta2[['Activitate', 'Cifra de afaceri']].to_csv('D:\subiecte_examen\subiect2\dateOUT\Cerinta2.csv')

# B1

x = pd.read_csv(filepath_or_buffer='D:\subiecte_examen\subiect2\dateIN\ProiectB.csv', index_col=0)
tinta = 'VULNERAB'
labels = list(x.columns[:-1].values)
print('Labels:', labels)

dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
x[tinta] = x[tinta].map(dict)
print(x[tinta])

x_train, x_test, y_train, y_test = train_test_split(x[labels], x[tinta], train_size=0.4)
model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train)
scores = model.transform(x_train)

pd.DataFrame(data=scores, columns=('z' + str(i+1) for i in range(scores.shape[1]))).to_csv('D:\subiecte_examen\subiect2\dateOUT\z.csv', index=False)

# B2

plt.figure(figsize=(12,12))
plt.title('Scoruri')
sb.kdeplot(scores, fill=True)
plt.show()

# B3

x_apply = pd.read_csv(filepath_or_buffer='D:\subiecte_examen\subiect2\dateIN\ProiectB_apply.csv', index_col=0)

prediction_test = model.predict(x_test)
prediction_applied = model.predict(x_apply)

pd.DataFrame(data=prediction_test).to_csv('D:\subiecte_examen\subiect2\dateOUT\predict_test.csv', index=False)
pd.DataFrame(data=prediction_applied).to_csv('D:\subiecte_examen\subiect2\dateOUT\predict_apply.csv', index=False)
