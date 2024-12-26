
```markdown
# Naive Bayes Text Classifier

A simple implementation of the Naive Bayes algorithm for text classification. This classifier is trained on a set of labeled documents and predicts the category of new, unseen documents.

## Features:
- Preprocessing of text: removes punctuation, converts to lowercase, and removes stop words.
- Classifies documents into categories based on training data using the Naive Bayes algorithm.

## How to Use:

### 1. **Training the Classifier:**

The classifier is trained on a set of labeled documents (text and their corresponding categories). Example training data is provided in the `docs` list.

```python
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
```

You can modify or add more documents to this list as needed.

### 2. Change the Dataset:

To train the classifier on a new dataset, follow these steps:

- **Modify `docs`:** Update or add new documents in the `docs` list. Each document should be a tuple with the format `('document_text', 'category')`.
- **Update Categories:** If you're adding new categories, make sure to include them in the `NaiveBayes` constructor. For example:

```python
nb = NaiveBayes(['category1', 'category2', 'category3'])
```

Ensure that the categories you pass to the constructor match the categories in your dataset.

### 3. **Making Predictions:**

After training, you can classify new documents:

```python
new_doc = 'I love playing football'
category = nb.predict(new_doc)
print(f'The document "{new_doc}" belongs to the category "{category}"')
```

## Requirements:
- Python 
- No additional libraries required.

## How to Run:
1. Clone this repository.
2. Modify the dataset in the `docs` list.
3. Run the script.

### Key Points:
- The README explains how to modify the dataset (`docs` list) to classify new text and categories.
- Instructions are simple, with clear examples of modifying the dataset and making predictions.
- It includes the required steps to get the code running with minimal setup.
