from flask import Flask, request, jsonify
from main import main
import json

# build the flask application as a server. Recieve the hint for image prcessing and process till the result finished.
app = Flask(__name__)


@app.route('/apply_ocr/', methods=['POST'])
def apply_ocr():
    # fetch the image url from the s3
    data = json.loads(request.data)

    url = str(data['url'])

    response = main(url)
    return response
