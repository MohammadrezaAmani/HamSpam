from read_file import create_dataset
from pre_process import preprocessing
from word2vec.tf_idf import tfidf
from word2vec.freq_word import freqword 
from word2vec.select_best_feature import selectKBest

from sklearn.model_selection import train_test_split

from models.knn import knn
from models.decision_tree import decision_tree
from models.logstic_regression import logstic_regression
from models.naive_bayes import naive_bayes
from models.random_forest import random_forest 

from plots.plots import create_plot

import warnings
warnings.filterwarnings('ignore')



df = create_dataset()
y = df["target"].values

pre_proc = preprocessing()
df = pre_proc.fit(df)

t_i = tfidf()
tf_idf_x = t_i.fit(df, "Text")
tf_idf_x = selectKBest(tf_idf_x, y)
tf_idf_x_train, tf_idf_x_test, tf_idf_y_train, tf_idf_y_test = train_test_split(tf_idf_x, y,random_state=0, test_size=0.2)


fw = freqword(df, "Text", "target")
freq_word_x = fw.transform()
freq_word_x_train, freq_word_x_test, freq_word_y_train, freq_word_y_test = train_test_split(freq_word_x, y,random_state=0, test_size=0.2)

tf_idf_acc = {
	"KNN" : None,
	"Logstic Regression" : None,
	"Decision Tree" : None,
	"Random Forest" : None,
	"Naive Bayes" : None,
}

freq_word_acc = {
	"KNN" : None,
	"Logstic Regression" : None,
	"Decision Tree" : None,
	"Random Forest" : None,
	"Naive Bayes" : None,
}

lg = logstic_regression()

lg.fit(tf_idf_x_train, tf_idf_y_train)
y_pred = lg.predict(tf_idf_x_test)
acc_lg = lg.accuracy(y_pred, tf_idf_y_test)
print(f"accuracy of logstic regression for tf-idf vec is {acc_lg * 100} %")
tf_idf_acc["Logstic Regression"] = acc_lg

lg.fit(freq_word_x_train, freq_word_y_train)
y_pred = lg.predict(freq_word_x_test)
acc_lg = lg.accuracy(y_pred, freq_word_y_test)
print(f"accuracy of logstic regression for freq-word vec is {acc_lg * 100} %")
freq_word_acc["Logstic Regression"] = acc_lg

KNN = knn()

KNN.fit(tf_idf_x_train, tf_idf_y_train)
y_pred = KNN.predict(tf_idf_x_test, 11)
acc_knn = KNN.accuracy(y_pred, tf_idf_y_test)
print(f"accuracy of knn for tf-idf vec is {acc_knn * 100} %")
tf_idf_acc["KNN"] = acc_knn

KNN.fit(freq_word_x_train, freq_word_y_train)
y_pred = KNN.predict(freq_word_x_test, 11)
acc_knn = KNN.accuracy(y_pred, freq_word_y_test)
print(f"accuracy of knn for freq-word vec is {acc_knn * 100} %")
freq_word_acc["KNN"] = acc_knn


dt = decision_tree(max_depth = 10)

dt.fit(tf_idf_x_train, tf_idf_y_train)
y_pred = dt.predict(tf_idf_x_test)
acc_dt = dt.accuracy(y_pred, tf_idf_y_test)
print(f"accuracy of decision tree for tf-idf vec is {acc_dt * 100} %")
tf_idf_acc["Decision Tree"] = acc_dt


dt.fit(freq_word_x_train, freq_word_y_train)
y_pred = dt.predict(freq_word_x_test)
acc_dt = dt.accuracy(y_pred, freq_word_y_test)
print(f"accuracy of decision tree for freq-word vec is {acc_dt * 100} %")
freq_word_acc["Decision Tree"] = acc_dt

rf = random_forest(n_trees = 5, max_depth = 10)

rf.fit(tf_idf_x_train, tf_idf_y_train)
y_pred = rf.predict(tf_idf_x_test)
acc_rf = rf.accuracy(y_pred, tf_idf_y_test)
print(f"accuracy of drandom forest for tf-idf vec is {acc_rf * 100} %")
tf_idf_acc["Random Forest"] = acc_rf


rf.fit(freq_word_x_train, freq_word_y_train)
y_pred = rf.predict(freq_word_x_test)
acc_rf = rf.accuracy(y_pred, freq_word_y_test)
print(f"accuracy of drandom forest for freq-word vec is {acc_rf * 100} %")
freq_word_acc["Random Forest"] = acc_rf


nb = naive_bayes()

nb.fit(tf_idf_x_train, tf_idf_y_train)
y_pred = nb.predict(tf_idf_x_test)
acc_nb = nb.accuracy(y_pred, tf_idf_y_test)
print(f"accuracy of naive bayes for tf-idf vec is {acc_nb * 100} %")
tf_idf_acc["Naive Bayes"] = acc_nb


nb.fit(freq_word_x_train, freq_word_y_train)
y_pred = nb.predict(freq_word_x_test)
acc_nb = nb.accuracy(y_pred, freq_word_y_test)
print(f"accuracy of naive bayes for feq-word vec is {acc_nb * 100} %")
freq_word_acc["Naive Bayes"] = acc_nb


create_plot(tf_idf_acc, freq_word_acc)
