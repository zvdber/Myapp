import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
#print(cancer_data.keys())
#print(cancer_data['DESCR'])
#cancer_data['data'].shape
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
print(df.head())
#print(cancer_data['target_names'])