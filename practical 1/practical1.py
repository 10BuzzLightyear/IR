import nltk
from nltk.corpus import stopwords


document1 = "The quick brown fox jumped over the lazy dog"
document2 = "The lazy dog slept in the sun"


nltk.download('stopwords')
stopWords = set(stopwords.words('english'))


tokens1 = document1.lower().split()
tokens2 = document2.lower().split()


terms = list(set(tokens1 + tokens2))


inverted_index = {}


for term in terms:
    if term in stopWords:
        continue  
    
    documents = []
    
    
    if term in tokens1:
        documents.append("Document 1")
    
    
    if term in tokens2:
        documents.append("Document 2")
    
    
    inverted_index[term] = documents


for term, documents in inverted_index.items():
    print(f"{term} ->", end=" ")
    
    for doc in documents:
        
        if doc == "Document 1":
            count_doc1 = tokens1.count(term)
            print(f"{doc} ({count_doc1}),", end=" ")
        else:
            count_doc2 = tokens2.count(term)
            print(f"{doc} ({count_doc2}),", end=" ")
    
    print()

# Add a footer for credit
print("\nPerformed by 740_Pallavi & 743_Deepak")
