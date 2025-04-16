from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

documents = [
    "Cats are known for their agility and grace",
    "Dogs are often called ‘man’s best friend’.",
    "Some dogs are trained to assist people with disabilities.",
    "The sun rises in the east and sets in the west.",
    "Many cats enjoy climbing trees and chasing toys."
]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

print(kmeans.labels_)
