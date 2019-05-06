from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib
from horsekickerpy.horsekick import clean_horse_kicks

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    df = pd.DataFrame(data)
    import pdb; pdb.set_trace()
    clean_df = clean_horse_kicks(df)
    model = joblib.load("results/horse_kick_model.pkl")

    return jsonify(model.predict(clean_df))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
