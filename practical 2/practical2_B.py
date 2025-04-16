    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
    import nltk
    from nltk.corpus import stopwords
    import numpy as np
    from numpy.linalg import norm

    train_set = ["The sky is blue.", "The sun is bright."]
    test_set = ["The sun in the sky is bright."]

    nltk.download('stopwords')
    stopWords = stopwords.words('english')

    vectorizer = CountVectorizer(stop_words=stopWords)
    transformer = TfidfTransformer()

    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(test_set).toarray()

    print('Training TF-IDF Array:', trainVectorizerArray)
    print('Test TF-IDF Array:', testVectorizerArray)

    cx = lambda a, b: round(np.inner(a, b) / (norm(a) * norm(b)), 3)

    for i, train_vec in enumerate(trainVectorizerArray):
        for test_vec in testVectorizerArray:
            cosine = cx(train_vec, test_vec)
            print(f"Cosine Similarity between Train Doc {i+1} and Test Doc: {cosine}")

    transformer.fit(trainVectorizerArray)
    print("\nTraining Set TF-IDF Representation:")
    print(transformer.transform(trainVectorizerArray).toarray())

    print("\nTest Set TF-IDF Representation:")
    tfidf = transformer.transform(testVectorizerArray)
    print(tfidf.todense())
