import sqlite3
from time import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


dat = sqlite3.connect("../../../../data/processed/test_train_database.db")
query = dat.execute("SELECT * From DryCleaned")
cols = [column[0] for column in query.description]
results = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

dataset = results.sample(frac=0.7)
devData = results.drop(dataset.index)

# Split the data into training/testing sets
X_train = dataset[["text"]]
X_test = devData[["text"]]
y_train = dataset[["isMergerOrAcquisition"]]
y_test = devData[["isMergerOrAcquisition"]]

"""
All improved hyper-parameters included in the below pipeline, no other parameters are set within the code
"""
# After Research, we will test Decision Trees, Naive Bayes, Support Vector Machines
pipeline = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                binary=True, lowercase=True, max_df=0.25, min_df=0, ngram_range=(1, 1)
            ),
        ),
        ("tfidf", TfidfTransformer(norm="l2", use_idf=False)),
        ("clf", SVC(C=1.5, gamma="scale", kernel="sigmoid")),
    ]
)

parameters = {
    # 'vect__lowercase': (True, False),
    # 'vect__min_df': (0, 0.025, 0.05, 0.1),
    # 'vect__max_df': (0.25, 0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams, bigrams, or trigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__C': (1.0, 1.5, 2.0, 2.5),
    # 'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'),
    # # 'clf__degree': (1, 2, 3, 4, 5),  # Uncomment if using poly kernel
    # 'clf__gamma': ('scale', 'auto'),
    # 'clf__shrinking': (True, False),
    # 'clf__probability': (True, False)
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
start = time()
grid_search = grid_search.fit(X_train.values.ravel(), y_train.values.ravel())

y_predicted = grid_search.predict(X_test.values.ravel())
#
# y_act = [i[0] for i in y_test.values]
# x_text = [i[0] for i in X_test.values]

# df = pd.DataFrame({
#     'actual': y_act,
#     'predicted': y_predicted,
#     'text': x_text})
#
# df['check'] = df['actual'] + df['predicted']

# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)

# print("*" * 10 + ' Failed Tests ' + "*" * 10)
# # df[df['check'] == 1].to_csv(r'data/failedTests.csv')
# print(df[df['check'] == 1])
# print("")
print("*" * 10 + " Metrics " + "*" * 10)
print(
    "GridSearchCV took %.2f seconds for %d candidate parameter settings."
    % (time() - start, len(grid_search.cv_results_["params"]))
)
report(grid_search.cv_results_)

print(
    metrics.classification_report(
        y_test.values.ravel(),
        y_predicted,
    )
)
print(
    "Mean squared error: %.2f" % mean_squared_error(y_test.values.ravel(), y_predicted)
)
print(" ")
print(metrics.confusion_matrix(y_test.values.ravel(), y_predicted))
print(" ")

# save the model
outfileX = "data/prop65DoesNotContain.pkl"

# TODO: ADD WAY TO PICKLE MODEL
# joblib.dump(grid_search.best_estimator_, outfileX)
# print("Pickle Complete.")

# TODO: Visualize the data
