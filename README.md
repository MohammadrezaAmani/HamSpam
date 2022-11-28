# Persian Spam email detector

This is a Persian spam email detector based on some machine learning algorithms. The project is written in [Python 3.11](https://python.org) and uses [Scikit-learn](https://scikit-learn.org/), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/) and [Hazm](https://github.com/sobhe/hazm) libraries.

## How to use

First, you need to install the required libraries. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

Then, you can open the "*[HamSpam.ipynb](https://github.com/MohammadrezaAmani/HamSpam/blob/main/src/HamSpam.ipynb)*" file in Jupyter Notebook and run the code.

## How it works

The project uses a dataset of 500 Persian spam emails and 500 Persian ham emails. The dataset is available in the "*[emails.csv](https://github.com/MohammadrezaAmani/HamSpam/blob/main/src/emails.csv)*" file. The dataset is divided into two parts: training and testing. The training part is used to train the model and the testing part is used to test the model. The model is trained using the following algorithms:

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
|[KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)| 0.736|
|[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)| 0.8|
|[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)| 0.944|
|[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)| 0.96|
|[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)| 0.948|
|[MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)| 0.94|
|[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)| 0.936|

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
