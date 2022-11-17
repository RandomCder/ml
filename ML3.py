import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Admission_Predict.csv')
print(df.columns)

from sklearn.preprocessing import Binarizer
bi=Binarizer(threshold=0.75)
df['Chance of Admit ']=bi.fit_transform(df[['Chance of Admit ']])
x=df.drop('Chance of Admit ',axis=1)
print(x)
y=df['Chance of Admit ']
print(y)
y=y.astype('int')
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.25)
print(x_train.shape)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
print(y_test)

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.show()
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
