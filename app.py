from flask import Flask, render_template, request
from model import AQIPredictor

app = Flask(__name__)

predictor = AQIPredictor("aqi.csv")

@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_names=predictor.feature_names
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_values = []

        for feature in predictor.feature_names:
            value = float(request.form[feature])
            input_values.append(value)

        result = predictor.predict_all(input_values)

        # Proper indentation 👇
        aqi = result["AQI_Prediction"]
        log_class = result["Logistic_Class"]
        knn_class = result["KNN_Class"]
        cluster = result["Cluster"]

        return render_template(
            "index.html",
            aqi=aqi,
            log_class=log_class,
            knn_class=knn_class,
            cluster=cluster,
            feature_names=predictor.feature_names
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            feature_names=predictor.feature_names
        )

if __name__ == "__main__":
    app.run(debug=True)