import pandas as pd
import numpy as np

test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')
test_passenger_ids = test_df['PassengerId']
train_df.info()
combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)

# Clean the data
combined_df['Age'].fillna(combined_df['Age'].median(), inplace=True)
combined_df['Fare'].fillna(combined_df['Fare'].median(), inplace=True)
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)
combined_df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True) 
combined_df['Sex'] = combined_df['Sex'].map({'male': 0, 'female': 1})
combined_df = pd.get_dummies(combined_df, columns=['Embarked'], drop_first=True)

combined_df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Split back into train and test sets
X_train = combined_df.iloc[:len(train_df), :]
X_test = combined_df.iloc[len(train_df):, :]
y_train = train_df['Survived']

print("\nData after cleaning:")
print(X_train.head())

