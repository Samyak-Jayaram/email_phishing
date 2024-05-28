import tkinter as tk
from tkinter import scrolledtext, messagebox
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("D:/Mini-project/mp-6th sem/MP_6th_sem/code/email.csv")
df = df.dropna()

email_type_counts = df['Email Type'].value_counts()
unique_email_types = email_type_counts.index.tolist()

color_map = {
    'Phishing Email': 'red',
    'Safe Email': 'green',
}

colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]

plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types with Custom Colors')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

Safe_Email = df[df["Email Type"] == "Safe Email"]
Phishing_Email = df[df["Email Type"] == "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])

Data = pd.concat([Safe_Email, Phishing_Email], ignore_index=True)

X = Data["Email Text"].values
y = Data["Email Type"].values

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

classifier = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", RandomForestClassifier(n_estimators=10))])
classifier.fit(X_train, y_train)

y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


def classify_email():
    email_text = email_entry.get("1.0", tk.END).strip()
    if not email_text:
        result_label.config(text="Prediction: Enter text", font=('Times New Roman', 18, 'bold'))
        root.after(30000, clear_prediction)
    else:
        prediction = classifier.predict([email_text])[0]
        result_label.config(text=f"Prediction: {prediction}", font=('Times New Roman', 18, 'bold'))
        root.after(30000, clear_prediction)


def clear_prediction():
    result_label.config(text="Prediction: ")


def display_metrics():
    metrics_text = f"Accuracy Score: {accuracy:.2f}\n\n"
    metrics_text += f"Confusion Matrix:\n{conf_matrix}\n\n"
    metrics_text += f"Classification Report:\n{class_report}"
    messagebox.showinfo("Model Metrics", metrics_text)


root = tk.Tk()
root.title("Phishing Email Classifier")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

email_label = tk.Label(frame, text="Enter Email Text:")
email_label.grid(row=0, column=0, pady=5)

email_entry = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=60, height=20)
email_entry.grid(row=1, column=0, pady=5)

classify_button = tk.Button(frame, text="Classify Email", font=('Times New Roman', 10, 'bold'), command=classify_email)
classify_button.grid(row=2, column=0, pady=5)

result_label = tk.Label(frame, text="Prediction: ", font=('Times New Roman', 18, 'bold'))
result_label.grid(row=3, column=0, pady=5)

metrics_button = tk.Button(frame, text="Show Model Metrics", font=('Times New Roman', 10, 'bold'), command=display_metrics)
metrics_button.grid(row=4, column=0, pady=5)

root.mainloop()
