# Tkinter is used for GUI but need to be changed to streamlit.
import tkinter as tk
from tkinter import scrolledtext, messagebox

# For Graph visualization.
import matplotlib.pyplot as plt

# To convert the given convert the text(feature) to vectors(`1s and 0s`) and fed to model 
from sklearn.feature_extraction.text import TfidfVectorizer

# To import the classification algorithm 
from sklearn.ensemble import RandomForestClassifier

# Chains together multiple processing steps into one streamline process
from sklearn.pipeline import Pipeline

# Datamanipulating library
import pandas as pd

# To split the data into training and testing data
from sklearn.model_selection import train_test_split

# For metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# To read the csv file
df = pd.read_csv("./email.csv")

# To drop the null values
df = df.dropna()



email_type_counts = df['Email Type'].value_counts()
unique_email_types = email_type_counts.index.tolist()

color_map = {
    'Phishing Email': 'red',
    'Safe Email': 'green',
}

colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]


# Visualisation of classification
plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types with Custom Colors')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# df == dataframes (Basically column name in a table)
Safe_Email = df[df["Email Type"] == "Safe Email"]
Phishing_Email = df[df["Email Type"] == "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])

# Merges dataframes together
Data = pd.concat([Safe_Email, Phishing_Email], ignore_index=True)

#X is the input , y is the output 
X = Data["Email Text"].values
y = Data["Email Type"].values

# Splitting the training data and testing data
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In this vectorization and classification runs together in the pipeline
classifier = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", RandomForestClassifier(n_estimators=10))])

# Fitting the teraining data into the model
classifier.fit(X_train, y_train)

# Taking the predictions
y_pred = classifier.predict(x_test)

# Model evaluation metrics 
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)



# Tkinter kelsa

def classify_email():
    # To get the email from the GUI 
    email_text = email_entry.get("1.0", tk.END).strip()
    
    # if Email text not entered
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
