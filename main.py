import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("student_data.csv")

# Features (input) and target (output)
X = data[["Hours", "Attendance", "Assignments"]]
y = data["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---- USER INPUT ----
print("\n--- Test Your Own Data ---")

h = float(input("Enter hours studied: "))
a = float(input("Enter attendance: "))
ass = float(input("Enter assignments completed: "))

# Fix warning (use DataFrame)
new_data = pd.DataFrame([[h, a, ass]], columns=["Hours", "Attendance", "Assignments"])

prediction = model.predict(new_data)

# Output result
if prediction[0] == 1:
    print("Prediction: PASS ✅")
else:
    print("Prediction: FAIL ❌")