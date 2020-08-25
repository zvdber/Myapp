import pandas as pd
df = pd.read_csv('titanic.csv')
small_df = df[['Age', 'Sex', 'Survived']]
df['male'] = df['Sex'].apply(lambda x:"male" if x== 'male' else "not male")
print(df[['male']]) 