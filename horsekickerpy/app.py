import pandas as pd
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource
from sklearn.externals import joblib

from horsekickerpy.horsekick import clean_horse_kicks
from horsekickerpy.wrapper import SMWrapper

api = Api()

app = Flask(__name__)
api.init_app(app)


@api.route('/predict')
class Predict(Resource):

    def post(self):
        data = request.get_json()
        df = pd.DataFrame(data)
        print(df)

        clean_df = clean_horse_kicks(df)

        model = joblib.load("model/horse_kick_model.pkl")

        return jsonify(list(model.predict(clean_df).round(3)))


@api.route('/test')
class Test(Resource):

    def get(self):
        return {"Hello": "World"}


#curl -H "Content-Type: application/json" --data @predict.json http://localhost:5000/predict
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
