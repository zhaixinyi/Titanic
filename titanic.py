import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from xgboost import XGBClassifier

train = pd.read_csv('/User/zxy/Desktop/Titanic')
test = pd.read_csv('/User/zxy/Desktop/Titanic')

selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']

X_train['Embarked'].fillna('S', inplace = True)
X_test['Embarked'].fillna('S', inplace = True)
X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace = True)

dict_vec = DictVectorizer(spare = False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient = 'record'))
print(dict_vec.feature_names_)

X_test = dict_vec.transform(X_test.to_dict(orient = 'record'))

rfc = RandomForestClassifier()
xgbc = XGBClassifier()
cross_val_score(rfc, X_train, y_train, cv = 5).mean()
cross_val_score(xgb, X_train, y_train, cv = 5).mean()

rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)
rfc_sub = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':rfc_predict})
rfc_sub.to_csv('rfc.csv', index = False)

xgbc.fit(X_train, y_train)
xgbc_predict = xgbc.predict(X_test)
xgbc_sub = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':xgbc_predict})
xgbc_sub.to_csv('xgbc.csv', index = False)


