import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import csv

# Loading df's and removing unwanted columns
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data = train_data.drop(['Cabin', 'Ticket', 'Name'], axis=1)
test_data = test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1)
test_data_label = pd.read_csv('gender_submission.csv')


# Filling in null Age values with Age mean
# Filling in Embarked na with most common embarked spot
train_mean_age = np.mean(train_data.Age)
train_data['Age'].fillna(train_mean_age, inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_mean_age = np.mean(test_data.Age)
test_data['Age'].fillna(test_mean_age, inplace=True)
test_mean_fare = np.mean(test_data.Fare)
test_data['Fare'].fillna(test_mean_fare, inplace=True)

#One-hot encoding, turning categoricals into binary representations
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

#Dropping survived from train_data and creating training labels
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

#train_test_split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#scaling the data
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.fit(X_test)

#fitting the LogisticRegression model
model = LogisticRegression()
model.fit(X_train, y_train)

#Scoring the model
train_score = model.score(X_test, y_test)
print(train_score)

#predicting the test_data
prediction = model.predict(test_data)
print(prediction)

csv_id_rows = [892]
id = 892
for i in range(892, 1309):
   id += 1
   csv_id_rows.append(id)

csv_value_rows = []
for value in prediction:
    csv_value_rows.append(value)

with open('Titanic_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['PassengerId', 'Survived'])
    for i in range(418):
        writer.writerow([csv_id_rows[i], csv_value_rows[i]])




