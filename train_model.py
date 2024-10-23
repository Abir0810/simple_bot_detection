import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Sample data structure (replace with real data)
data = {
    'mouse_speed': [50, 120, 45, 110, 90],  # example mouse speeds
    'keypress_interval': [0.5, 0.3, 0.9, 0.7, 0.2],  # example time between keypresses
    'task_time': [10, 3, 12, 6, 4],  # total time taken to submit the form
    'label': ['human', 'bot', 'human', 'bot', 'bot']  # labels for training
}

df = pd.DataFrame(data)

# Features and labels
X = df[['mouse_speed', 'keypress_interval', 'task_time']]
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model
with open('bot_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)