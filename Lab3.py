# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Step 1: Data Collection
# Load the Titanic dataset from seaborn
data = sns.load_dataset('titanic')

# Step 2: Data Cleaning
# Check for missing values
print("Missing values before cleaning:\n", data.isnull().sum())

# Drop columns with excessive missing values
data.drop(columns=['deck'], inplace=True)

# Impute missing values
data['age'].fillna(data['age'].median(), inplace=True)
data['embark_town'].fillna(data['embark_town'].mode()[0], inplace=True)

print("Missing values after cleaning:\n", data.isnull().sum())

# Step 3: Handling Outliers
# Visualize outliers
sns.boxplot(data['fare'])
plt.title("Boxplot for Fare")
plt.show()

sns.boxplot(data['age'])
plt.title("Boxplot for Age")
plt.show()

# Handle outliers in 'fare' using capping
Q1 = data['fare'].quantile(0.25)
Q3 = data['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data['fare'] = data['fare'].clip(lower_bound, upper_bound)

# Step 4: Data Normalization
scaler = MinMaxScaler()
data[['age', 'fare']] = scaler.fit_transform(data[['age', 'fare']])

# Step 5: Feature Engineering
# Create new features
data['family_size'] = data['sibsp'] + data['parch']
data['title'] = data['name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Fill missing values in new features for simplicity (if any)
data.fillna(0, inplace=True)

# Step 6: Feature Selection
# Correlation analysis
corr_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature importance using RandomForest
features = ['age', 'fare', 'family_size']
target = 'survived'

X = data[features]
y = data[target]

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
feature_importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
print("Feature Importances:\n", feature_importances)

# Step 7: Model Building
# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
