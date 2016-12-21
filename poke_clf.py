from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

df = pd.read_csv('Pokemon.csv')
del df['#']
del df['Name']
del df['Type 2']
del df['Legendary']
del df['Total']


y = df['Type 1'].values

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)


class PokemonClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, poly=2, n_estimators=50):
        super(BaseEstimator, self).__init__()
        self.poly = poly
        self.n_estimators = n_estimators

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            return None
        if 'Generation' in X.columns:
            self.classes_ = unique_labels(y)
            self.clfs_ = {}
            for index, group in X.groupby('Generation'):
                clf = RandomForestClassifier(self.n_estimators)
                x = group.drop(['Generation', 'Type 1'], axis=1)
                y = group['Type 1']
                poly = PolynomialFeatures(self.poly)
                x = poly.fit_transform(x)
                clf.fit(x, y)
                self.clfs_[index] = clf
            return self
        return None

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            return [None for each in range(len(X))]
        result = []
        if 'Generation' in X.columns:
            poly = PolynomialFeatures(self.poly)
            gen = X['Generation']
            x = X.drop(['Generation', 'Type 1'], axis=1)
            x = poly.fit_transform(x)
            for g, each in zip(gen, x):
                clf = self.clfs_[g]
                each = each.reshape(1, -1)
                result.append(clf.predict(each)[0])
        return result if result else [None for _ in range(len(X))]


t = PokemonClassifier(poly=3, n_estimators=120)
cross_val_score(t, x_train, y_train, cv=3)
t.fit(x_train, y_train)
y_pred = t.predict(x_test)
accuracy_score(y_test, y_pred)
