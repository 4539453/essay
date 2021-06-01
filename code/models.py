import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import sys
import gensim
import numpy as np
from scipy import interp
from sklearn.pipeline import Pipeline
from sklearn import metrics, svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

W2V_FILE = "model.txt"
ELMO_FILE = '199.zip'
FOLDS = 10

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec)

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def load_data_and_labels(data):
    x_text = [t.split() for t in data.sentence.values]
    labels = data.target.values
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return x_text,labels

def preprocess_w2v(X, y):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=FOLDS)
    for train_index, test_index in skf.split(X, y):
        X_train_value, X_test_value = X[train_index], X[test_index]
        y_train_value, y_test_value = y[train_index], y[test_index]
        X_train.append(X_train_value)
        X_test.append(X_test_value)
        y_train.append(y_train_value)
        y_test.append(y_test_value)
    return X_train, X_test, y_train, y_test

def preprocess_elmo(X, y):
    emb_model = ElmoModel()
    emb_model.load(ELMO_FILE)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=FOLDS)
    for train_index, test_index in skf.split(X, y):
        features_train, features_test = X[train_index], X[test_index]
        t_labels_train, t_labels_test = y[train_index], y[test_index]
        features_train = elmo.get_elmo_vector_average(features_train)
        features_test  = elmo.get_elmo_vector_average(features_test)
        selector = SelectPercentile(f_classif, percentile=10)
        selector.fit(features_train, t_labels_train)
        X_train.append(selector.transform(features_train).toarray())
        X_test.append(selector.transform(features_test).toarray())
        y_train.append(t_labels_train)
        y_test.append(t_labels_test)
    return X_train, X_test, y_train, y_test

def train(clf, X_train, X_test, y_train, y_test):
    headers = 'fl,precision,recall,f1,support,acc,TN, FP, FN, TP'
    print(headers)
    for fl in range(FOLDS):
        clf.fit(X_train[fl], y_train[fl])
        pred = clf.predict(X_test[fl])
        print(metrics.classification_report(y_test[fl], pred))
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_test[fl], pred, average="samples")
        support = conf_matrix[0][0]+conf_matrix[0][1]+conf_matrix[1][0]+conf_matrix[1][1]
        print(y_test[fl])
        print(pred)
        roc_auc = roc_auc_score(y_test[fl], pred, multi_class='ovo')
        print(f'roc_auc: {roc_auc}')
        print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (fl, precision, recall, f1, support, acc, conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]))

def benchmark(input_file):
    X, y = load_data_and_labels(data)

    clfs = [  
              RandomForestClassifier(class_weight="balanced"),
              svm.SVC(),
              LogisticRegression(solver="liblinear")
              
    ]

    names = ['RandomForest', 'SVM', 'log reg']

    print("WORD2VEC")
    with open(W2V_FILE, "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    X_train, X_test, y_train, y_test = preprocess_w2v(X,y)
    cont = 0
    sections = ['Word2Vec', 'ELMo']

    results = {}
    section = sections[cont]
    print(section)
    results[section] = {}
    for i in range(len(names)):
        print(section + " " + names[i])
        clf = Pipeline([
            (section + " vectorizer", MeanEmbeddingVectorizer(w2v)),
            ("classifier", clfs[i])])
        results[section][names[i]] = train(clf, X_train, X_test, y_train, y_test)
    cont += 1
    section = sections[cont]
    print(section)
    results[section] = {}
    X_train, X_test, y_train, y_test = preprocess_elmo(X,y)
    for i in range(len(names)):
        print(section + " " + names[i])
        clf = clfs[i]
        results[section][names[i]] = train(clf, X_train, X_test, y_train, y_test)
    for name in names:
        records = []
        for section in sections:
            values = results[section][name]
            values['label'] = section
            records.append(values)

def run(params):
    if params["bench"]:
        benchmark(params)
    else:
        predict(params)

def main(argv):
    params = {
        "bench": True,
        "w2v_file": W2V_FILE
    }
 
    run(params)

if __name__ == "__main__":
    main(sys.argv[1:])