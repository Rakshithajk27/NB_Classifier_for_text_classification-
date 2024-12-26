import re
from collections import defaultdict

class NaiveBayes:
    def __init__(self, classes):
        self.classes = classes
        self.vocab = set()
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_doc_counts = defaultdict(int)
        self.num_docs = 0

    def preprocess(self, text):
        # Remove punctuations and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text).lower()
        # Remove stop words
        stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'of', 'to', 'for', 'by', 'with', 'from', 'and'])
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def train(self, documents):
        for document, category in documents:
            tokens = self.preprocess(document)
            self.vocab.update(tokens)
            self.class_doc_counts[category] += 1
            self.num_docs += 1
            for word in tokens:
                self.class_word_counts[category][word] += 1

    def predict(self, document):
        tokens = self.preprocess(document)
        posteriors = {category: 0 for category in self.classes}
        for category in self.classes:
            prior = self.class_doc_counts[category] / self.num_docs
            posterior = prior
            for word in tokens:
                word_count = self.class_word_counts[category][word]
                total_count = sum(self.class_word_counts[category].values())
                conditional = word_count / total_count if total_count > 0 else 1  # Avoid division by zero
                posterior *= conditional
            posteriors[category] = posterior
        return max(posteriors, key=posteriors.get)


# Training data
docs = [
    ('I like pizza', 'fast food'),
    ('lets play football', 'sports'),
    ('lets go study', 'college'),
    ('lets start python coding', 'coding'),
    ('lets ride the bike', 'riding'),
    ('I love biryani', 'food'),
    ('Its raining', 'weather'),
    ('The car is fast', 'vehicle'),
    ('The election is coming up', 'politics'),
]

# Instantiate and train the Naive Bayes classifier
nb = NaiveBayes(['fast food', 'sports', 'college', 'coding', 'riding', 'food', 'weather', 'vehicle', 'politics'])
nb.train(docs)

# Predict the category of a new document
new_doc1 = 'I Like Pasta'
new_doc2 = 'lets play football'
new_doc3 = 'lets go study'

category1 = nb.predict(new_doc1)
category2 = nb.predict(new_doc2)
category3 = nb.predict(new_doc3)

# Output the results
print(f'The document "{new_doc1}" belongs to the category "{category1}"')
print(f'The document "{new_doc2}" belongs to the category "{category2}"')
print(f'The document "{new_doc3}" belongs to the category "{category3}"')
