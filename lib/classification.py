from landingai.predict import Predictor

def get_classification(image, api_key, endpoint_id):
    predictor = Predictor(endpoint_id, api_key=api_key)
    predictions = predictor.predict(image)

    if len(predictions) == 0:
        return {}

    primary_prediction = predictions[0]
    prediction_json = primary_prediction.json()
    return prediction_json
