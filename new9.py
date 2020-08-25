from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X=[]
X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'Age', 'male']].values
X3 = df[['Age', 'Fare']].values
y = df['Survived'].values

kf = KFold(n_splits=5, shuffle=True)

model1 = LogisticRegression()
model2 = LogisticRegression()
model3 = LogisticRegression()

scores = []

splits = list(kf.split(X1))
scores = cross_val_score(model1, X1, y, cv=5)
print(scores)
print(scores.mean())

splits = list(kf.split(X2))
scores = cross_val_score(model2, X2, y, cv=5)
print(scores)
print(scores.mean())

splits = list(kf.split(X3))
scores = cross_val_score(model3, X3, y, cv=5)
print(scores)
print(scores.mean())
