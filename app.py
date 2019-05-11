import pandas as pd
from flask import Flask, jsonify, request
from sklearn.externals import joblib

from horsekickerpy.horsekick import clean_horse_kicks
from horsekickerpy.wrapper import SMWrapper

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    df = pd.DataFrame(data)
    clean_df = clean_horse_kicks(df)

    model = joblib.load("results/horse_kick_model.pkl")

    return jsonify(list(model.predict(clean_df).round(3)))


@app.route('/test', methods=['POST'])
def test():
    return jsonify(request.get_json())


#curl -H "Content-Type: application/json" --data @predict.json http://localhost:5000/predict
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
