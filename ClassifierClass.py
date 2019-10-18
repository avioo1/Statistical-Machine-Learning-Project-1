from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import pickle


class ClassifierClass:

    def __init__(self, clf=None, vm=None, vm2=None):

        if clf:
            self.clf = clf
        else:
            self.clf = MLPClassifier(hidden_layer_sizes=(350,), max_iter=5, alpha=1e-4,
                                     solver='adam', verbose=5, tol=1e-4, random_state=1,
                                     early_stopping=True)
            # self.clf = XGBClassifier(verbosity=2)
        self.vm = vm
        self.vm2 = vm2
        # self.clf.out_activation_ = 'softmax'

    def fit(self, X, y, run_cross_val=False):
        self.clf.fit(X, y)
        if run_cross_val:
            print(cross_val_score(self.clf, X, y, cv=3, n_jobs=-1))

    def predict(self, X):
        return self.clf.predict(X)

    def save(self, filename):
        self.__module__ = "ClassifierClass"
        f = open(filename, "wb")
        pickle.dump(self, f)
        f.close()
