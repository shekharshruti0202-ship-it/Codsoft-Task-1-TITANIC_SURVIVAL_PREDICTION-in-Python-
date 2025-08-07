import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
train = pd.read_csv(r"C:\Users\shekh\OneDrive\TITANIC_SURVIVAL_PREDICTION\train.csv")
test = pd.read_csv(r"C:\Users\shekh\OneDrive\TITANIC_SURVIVAL_PREDICTION\test.csv")
test_passenger_ids = test["PassengerId"]

full_data = pd.concat([train, test], sort=False)

full_data['Title'] = full_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
full_data['Title'] = full_data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
full_data['Title'] = full_data['Title'].replace('Mlle', 'Miss')
full_data['Title'] = full_data['Title'].replace('Ms', 'Miss')
full_data['Title'] = full_data['Title'].replace('Mme', 'Mrs')

full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
full_data['IsAlone'] = 0
full_data.loc[full_data['FamilySize'] == 1, 'IsAlone'] = 1

full_data['Age']=full_data['Age'].fillna(full_data['Age'].median(), inplace=True)
full_data['Embarked']=full_data['Embarked'].fillna(full_data['Embarked'].mode()[0], inplace=True)
full_data['Fare']=full_data['Fare'].fillna(full_data['Fare'].median(), inplace=True)

full_data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

full_data = pd.get_dummies(full_data, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

train = full_data[:len(train)]
test = full_data[len(train):].drop(['Survived'], axis=1)
X = train.drop(['Survived'], axis=1)
y = train['Survived']
X_test = test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
test_predictions = model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": test_predictions
})
submission.to_csv("titanic_submission_improved.csv", index=False)

print("Submission file created: titanic_submission_improved.csv")

