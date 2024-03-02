import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

"""
для задачі логістичної регресії виконати наступне
1) розбиття датасету на train/cross_val/test
2) нормалізація даних
    2.1) тренування базової моделі (model1) лише на трейн датасеті 
3) використати cross_val датасет для підбору оптимального значення якогось гіперпараметру (learning rate чи степінь полінома чи тд)
3) обчислити тестові метрики для кращої отриманої моделі і порівняти їх із метриками для моделі model1. пояснити отримані результати 
"""

data = pd.read_csv('diabetes2.csv')

# first five row
print(f'First five row:\n {data.head()}')
print('\n')

# info
print(f'Info:\n{data.info()}')
print('\n')

# describe
print(f'Describe:\n{data.describe()}')
print('\n')

# count space in column
print(f'Count space in column:\n{data.isnull().sum()}')
print('\n')

# max in column
print(f'Max value of column:\n{data.max()}')
print('\n')

# min in column
print(f'Min value of column:\n{data.min()}')
print('\n')

# count of unique in column
print(f'Count of unique values in column:\n{data.nunique()}')
print('\n')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split dataset
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

# feature normalization
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

# train base model1
model1 = LogisticRegression(random_state=0, max_iter=1000).fit(X_train_sc, y_train)

# predict test set
predict = model1.predict(X_test_sc)

# accuracy for basic model
accuracy = accuracy_score(y_test, predict)
print(f'Accuracy basic model: {accuracy} %')

# using cross validation
alpha = [0.001, 0.01, 0.1, 1, 10]
val_accuracy = []

for i in alpha:
    model = LogisticRegression(C=i, random_state=0)
    val_score = cross_val_score(model, X_val_sc, y_val, cv=5, scoring="accuracy")
    val_accuracy.append(np.mean(val_score))

optimal_alpha = alpha[np.argmax(np.array(val_accuracy))]

# creating model with best parameter
best_model = LogisticRegression(C=optimal_alpha, random_state=0).fit(X_train_sc, y_train)

# predict with best parameter
predict_val = best_model.predict(X_test_sc)

# accuracy CV best parameter
accuracy_val = accuracy_score(y_test, predict_val)
print(f'Accuracy with optimal alpha: {accuracy_val} %')

# comparison of parameter
if accuracy > accuracy_val:
    print('Basic model is better')
elif accuracy < accuracy_val:
    print('CV with optimal alpha is better')
else:
    print('Results equal')

print('\n-----------------------------------------------------------------------------------------------------------')
print('Using LogisticRegressionCV()')
# using LogisticRegressionCV()
model_cv = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10], cv=5, random_state=0).fit(X_val_sc, y_val)

# optimal alpha
optimal_alpha_cv = model_cv.C_[0]
print('Optimal alpha: ', optimal_alpha_cv)

# creating model with best parameter
best_model_cv = LogisticRegression(C=optimal_alpha_cv, random_state=0).fit(X_train_sc, y_train)

# predict with best parameter
predict_cv = best_model_cv.predict(X_test_sc)

# accuracy with best parameter
accuracy_cv = accuracy_score(y_test, predict_cv)
print(f'Accuracy basic model: {accuracy} %')
print(f'Accuracy using LogisticRegressionCV(): {accuracy_cv} %')

# comparison of parameter
if accuracy > accuracy_cv:
    print('Basic model is better')
elif accuracy < accuracy_cv:
    print( 'CV with optimal alpha(LogisticRegressionCV) is better')
else:
    print('Results equal')
