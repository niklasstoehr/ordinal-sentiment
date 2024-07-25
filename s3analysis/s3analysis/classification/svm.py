from sklearn.svm import SVC

class SVM():

    def __init__(self, task_type, data_type, seq = 1, n_features = 1):
        self.model = SVC()
        print(f"SVM on {task_type} with {data_type} seq {seq} n_features {n_features}")

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

