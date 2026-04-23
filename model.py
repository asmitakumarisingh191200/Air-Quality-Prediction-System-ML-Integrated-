import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

class AQIPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.lr = LinearRegression()
        self.log_model = LogisticRegression()
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.kmeans = KMeans(n_clusters=3, random_state=42)

        self.feature_names = None
        self.train_models()

    def train_models(self):
        data = pd.read_csv(r"C:\Users\asmit\Downloads\Python\aqi.csv")
        data = data.dropna()

        numeric_data = data.select_dtypes(include=[np.number])

        X = numeric_data.iloc[:, :-1]
        y = numeric_data.iloc[:, -1]

        self.feature_names = X.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)

        # Regression
        X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.lr.fit(X_train, y_train)

        # Classification
        y_class = (y > y.median()).astype(int)
        X_train_c, _, y_train_c, _ = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

        self.log_model.fit(X_train_c, y_train_c)
        self.knn.fit(X_train_c, y_train_c)

        # Clustering
        self.kmeans.fit(X_scaled)

    def predict_all(self, input_values):
        input_array = np.array(input_values).reshape(1, -1)
        scaled_input = self.scaler.transform(input_array)

        lr_pred = self.lr.predict(scaled_input)[0]
        log_pred = self.log_model.predict(scaled_input)[0]
        knn_pred = self.knn.predict(scaled_input)[0]
        cluster = self.kmeans.predict(scaled_input)[0]

        return {
            "AQI_Prediction": round(lr_pred, 2),
            "Logistic_Class": int(log_pred),
            "KNN_Class": int(knn_pred),
            "Cluster": int(cluster)
        }