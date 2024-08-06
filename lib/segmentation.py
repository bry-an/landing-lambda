from landingai.predict import Predictor
from landingai.visualize import overlay_bboxes

def get_segmentation(image, api_key, endpoint_id):
    predictor = Predictor(endpoint_id, api_key=api_key)
    predictions = predictor.predict(image)

    if len(predictions) == 0:
        return ({}, image)

    primary_prediction = predictions[0]
    bounding_boxes = primary_prediction.bboxes
    segmentation_result_overlay = overlay_bboxes([primary_prediction], image, {"draw_label": False})
    prediction_json = primary_prediction.json()
    return (prediction_json, segmentation_result_overlay)
