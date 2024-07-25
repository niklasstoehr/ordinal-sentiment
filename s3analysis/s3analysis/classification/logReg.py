from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class Logistic_Regression():

    def __init__(self, task_type, data_type, seq = 1, n_features=1):
        #self.lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        #self.model = make_pipeline(StandardScaler(), self.lr)
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        print(f"log regression on {task_type} with {data_type} seq {seq} n_features {n_features}")

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred


