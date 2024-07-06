import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "package"))
from lib.segmentation import get_segmentation
from lib.classification import get_classification
from io import BytesIO
from PIL import Image

import json
import base64
from requests_toolbelt.multipart import decoder, encoder

def segmentation_handler(event, context):
    try:
        body = base64.b64decode(event["body"])

        content_type = event["headers"]["content-type"]

        if not content_type.startswith('multipart/form-data'):
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Invalid content type' + content_type})
            }

        multipart_data = decoder.MultipartDecoder(body, content_type)

        api_key = None
        image_content = None
        endpoint_id = None

        for part in multipart_data.parts:
            content_disposition = part.headers[b'Content-Disposition'].decode('utf-8')
            if 'filename' in content_disposition:
                image_content = part.content
            else:
                field_name = content_disposition.split('name="')[1].split('"')[0]
                if field_name == 'api_key':
                    api_key = part.text
                elif field_name == 'endpoint_id':
                    endpoint_id = part.text

        if not api_key or not image_content or not endpoint_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing data'})
            }

        image = Image.open(BytesIO(image_content))

        # perform inference
        prediction, segmentation_result_overlay = get_segmentation(image, api_key, endpoint_id)

        image_io = BytesIO()
        segmentation_result_overlay.save(image_io, format='JPEG')
        image_io.seek(0)

        fields = {
            'predictions': ('prediction.json', json.dumps(prediction), 'application/json'),
            'image': ('segmentation_result_overlay.jpeg', image_io, 'image/jpeg')
        }

        m = encoder.MultipartEncoder(fields)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': m.content_type
            },
            'body': base64.b64encode(m.to_string()).decode('utf-8'),
            'isBase64Encoded': True
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def classification_handler(event, context):
    try:
        body = base64.b64decode(event["body"])

        content_type = event["headers"]["content-type"]

        if not content_type.startswith('multipart/form-data'):
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Invalid content type' + content_type})
            }

        multipart_data = decoder.MultipartDecoder(body, content_type)

        api_key = None
        image_content = None
        endpoint_id = None

        for part in multipart_data.parts:
            content_disposition = part.headers[b'Content-Disposition'].decode('utf-8')
            if 'filename' in content_disposition:
                image_content = part.content
            else:
                field_name = content_disposition.split('name="')[1].split('"')[0]
                if field_name == 'api_key':
                    api_key = part.text
                elif field_name == 'endpoint_id':
                    endpoint_id = part.text

        if not api_key or not image_content or not endpoint_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing data'})
            }

        image = Image.open(BytesIO(image_content))

        # perform inference
        prediction = get_classification(image, api_key, endpoint_id)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'prediction': prediction})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
