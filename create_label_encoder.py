# create_label_encoder.py

from sklearn.preprocessing import LabelEncoder
import pickle

# Define your classes
labels = ['Spam', 'ham']

# Fit LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Save to file
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ label_encoder.pkl saved successfully.")
