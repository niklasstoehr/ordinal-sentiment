from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class XG_Boost():

    def __init__(self, task_type, data_type, seq = 1, n_features = 1):
        self.clf = XGBClassifier()
        self.model = make_pipeline(StandardScaler(), self.clf)
        print(f"xgBoost on {task_type} with {data_type} seq {seq} n_features {n_features}")


    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred


