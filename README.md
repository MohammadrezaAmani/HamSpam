# Persian Spam email detector

This is a Persian spam email detector based on some machine learning algorithms. The project is written in Python 3.11 and uses Scikit-learn, Numpy, Pandas and Hazm libraries.

## How to use

First, you need to install the required libraries. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

Then, you can open the `HamSpam.ipynb` file in Jupyter Notebook and run the code.

## How it works

The project uses a dataset of 500 Persian spam emails and 500 Persian ham emails. The dataset is available in the `emails.csv` file. The dataset is divided into two parts: training and testing. The training part is used to train the model and the testing part is used to test the model. The model is trained using the following algorithms:

- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Logistic Regression
- and ...
with accuracy of 94.8%.

## accuracy of models:
| Model | Accuracy |
|--|--|
|KNeighborsClassifier| 0.736|
|DecisionTreeClassifier| 0.8|
|RandomForestClassifier| 0.944|
|LogisticRegression| 0.96|
|SGDClassifier| 0.948|
|MultinomialNB| 0.94|
|SVC| 0.936|

## How to improve accuracy

increase the number of features. this operation nees more time to compute but get accuracy about 98%.

replace
```python3
word_features = list(all_words.keys())[:1500]
```
with:
```python3
word_features = list(all_words.keys())
```
