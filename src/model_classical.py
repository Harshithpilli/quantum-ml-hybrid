from sklearn.ensemble import RandomForestClassifier


class ClassicalModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)


    def train(self, X, y):
        self.model.fit(X, y)


    def predict(self, X):
        return self.model.predict(X)


    def save(self, path):
        import joblib
        joblib.dump(self.model, path)


    def load(self, path):
        import joblib
        self.model = joblib.load(path)