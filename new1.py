#import pandas as pd
#df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
#print(df.shape)
#arr=df[['Survived', 'Fare', 'Age']].values[1:10]
#mask = arr[:, 2] < 18
#print(arr[mask])
#print(arr[arr[:, 2] < 18])
#print((arr[:, 2] < 18).sum())

import matplotlib.pyplot as plt
#plt.scatter(df['Fare'], df['Age'], c=df['Survived'])
#plt.plot([0, 80], [85, 5])

plt.scatter(20,50)
plt.scatter(100,20)
plt.xlabel("Fare")
plt.ylabel("Age")
plt.plot([30,110],[0,80])
plt.xlim((0,500))
plt.ylim((0,80))

plt.colorbar()
plt.show()