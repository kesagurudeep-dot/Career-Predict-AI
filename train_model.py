import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv("student_placement_dataset.csv")

# Convert text columns into numbers
le = LabelEncoder()

data["Branch"] = le.fit_transform(data["Branch"])
data["Python"] = le.fit_transform(data["Python"])
data["Java"] = le.fit_transform(data["Java"])
data["SQL"] = le.fit_transform(data["SQL"])
data["Internship"] = le.fit_transform(data["Internship"])
data["Company"] = le.fit_transform(data["Company"])

# Features and Target
X = data[["Branch", "CGPA", "Python", "Java", "SQL",
          "Projects", "Internship", "Certifications"]]
y = data["Company"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")