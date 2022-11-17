import pandas as pd
df=pd.read_csv('Heart.csv')

print(df.shape)

print(df.isnull())
print(df.isnull().sum())
print(df.isnull().sum().sum())

print(df.dtypes)

print(df[df==0].count())

print(df['Age'].mean())

print(df[['Age','Sex']])

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,random_state=0,test_size=0.25)
print(train.shape)
print(test.shape)
