import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel('hairfall_problem3592.xlsx')

# Drop unnecessary columns
df.drop(columns=['Timestamp', 'What is your name ?'], inplace=True)

# Rename target column for ease
df.rename(columns={'Do you have hair fall problem ?': 'Hair_Fall_Problem'}, inplace=True)

# Encode categorical columns and store encoders
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = df[column].astype(str)  # Convert to string before encoding
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Save encoders with the model
model = GradientBoostingClassifier()
X = df.drop('Hair_Fall_Problem', axis=1)
y = df['Hair_Fall_Problem']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and encoders together
joblib.dump((model, encoders), 'model/hair_loss_model.pkl')  # Make sure 'model/' folder exists

# ------------------- Plots & Analysis ------------------------

plt.style.use('ggplot')
sns.set_theme(style='whitegrid')
sns.set(rc={'figure.figsize':(10,6)})

# 1. Target distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Hair_Fall_Problem', data=df, palette='viridis')
plt.title('Distribution of Hair Fall Problem')
plt.xlabel('Hair Fall Problem (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Print model features
print("Model features:", df.drop('Hair_Fall_Problem', axis=1).columns.tolist())
