import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

try:
    df = pd.read_csv(r"C:\Users\avina\OneDrive\Documents\sem6_practical\Dataset.csv")
except FileNotFoundError:
    print("Error: Dataset.csv file not found!")
    exit()

data = df["covid"] + " " + df["fever"]
X = data.astype(str)
y = df['flu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

try:
    data1 = pd.read_csv(r"C:\Users\avina\OneDrive\Documents\sem6_practical\Test.csv")
except FileNotFoundError:
    print("Error: Test.csv file not found!")
    exit()

new_data = data1["covid"] + " " + data1["fever"]
new_data_counts = vectorizer.transform(new_data.astype(str))

predictions = classifier.predict(new_data_counts)

print("Predictions for new data:")
print(predictions)

accuracy = accuracy_score(y_test, classifier.predict(X_test_counts))
print(f"\nAccuracy: {accuracy:.2f}")
print("Classification Report: ")
print(classification_report(y_test, classifier.predict(X_test_counts), zero_division=0))

predictions_df = pd.DataFrame(predictions, columns=['flu_prediction'])
data1 = pd.concat([data1, predictions_df], axis=1)

output_path = r"C:\Users\avina\Documents\Sem 6\IR\Test1.csv"

# Create the directory if it doesn't exist
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data1.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")
