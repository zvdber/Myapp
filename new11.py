import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# Loading the iris data-set
iris = load_iris()
data, target, fns = (
    iris['data'], iris['target'], iris['feature_names']
)

# k_fold cross validation
k_fold = KFold(shuffle=True, random_state=1)
params = {
    'k_fold': list(k_fold.split(data)),
    'features': [data, data[:, 1:], data[:, 2:], data[:, 3:], data[:, :-1]],
    'feature_names': [fns, fns[1:], fns[2:], fns[3:], fns[:-1]]
}

count = 1

for _data, f, fn in zip(*params.values()):
    # train test data indices
    train_indices, test_indices = _data

    # spliting data into train test set
    x_train, x_test, y_train, y_test = (
        f[train_indices], f[test_indices], target[train_indices], target[test_indices]
    )

    print(f"models {count} with {f.shape[1]} feature(s)", *fn, sep=' | ')

    model = LogisticRegression(max_iter=500, random_state=0)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy, precision, recall, f1 = (
        accuracy_score(y_test, y_pred), np.mean(precision_score(y_test, y_pred, average=None)),
        np.mean(recall_score(y_test, y_pred, average=None)), np.mean(f1_score(y_test, y_pred, average=None))
    )

    count += 1
    print(f"""accuracy: {round(accuracy, 4)}, precision: {round(precision, 4)},
recall: {round(recall, 4)}, f1_score: {round(f1, 4)}""")
    print("#################################", end="\n\n")
