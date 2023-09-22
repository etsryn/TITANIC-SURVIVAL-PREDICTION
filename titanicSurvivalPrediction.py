import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns


# Load the Titanic dataset
titanic_data = pd.read_csv("titanic_dataset.csv")

# Display the first few rows of the dataset
# print(titanic_data.head())

# Check for missing values
missing_values = titanic_data.isnull().sum()
# print("Missing Values : ",missing_values)

# Visualize the data (e.g., using seaborn)
sns.countplot(x='Survived', data=titanic_data)
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
os.system('cls')
# Add more visualizations as needed
plt.show()

# Handle missing values (e.g., fill missing ages with the mean age)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# Fill missing values in the 'Embarked' column with the mode
mode_embarked = titanic_data['Embarked'].mode()[0]
titanic_data['Embarked'].fillna(mode_embarked, inplace=True)

# Fill missing values in the 'Fare' column with the mean fare
titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)

# Encode categorical variables (e.g., 'Sex' and 'Embarked' columns)
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'])

# Drop irrelevant columns or features with too many missing values
titanic_data = titanic_data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

# Scale numerical features (e.g., 'Age' and 'Fare')
scaler = StandardScaler()
titanic_data[['Age', 'Fare']] = scaler.fit_transform(titanic_data[['Age', 'Fare']])

# Split the data into training and testing sets
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']
# Check if there are any remaining NaN values in the dataset
if X.isnull().values.any():
    print("There are still NaN values in the dataset. Please check and handle them.")
else:
    # Create and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Display a classification report
    print(classification_report(y_test, y_pred))

    # Visualize a confusion matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.show()