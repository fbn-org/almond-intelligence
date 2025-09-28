import base64
import os

from flask import Flask, Response, after_this_request, jsonify, request
from flask_cors import CORS
from inference import predict

app = Flask(__name__)
CORS(
    app,
    origins="http://localhost:5173",
    supports_credentials=True,
    methods=["GET", "POST", "OPTIONS"],
)


@app.route("/almond", methods=["POST", "GET"])
def almond():
    @after_this_request
    def add_header(response):
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    image_data = request.get_json().get("image", None)
    if image_data.startswith("data:image/png;base64,"):
        image_data = image_data.replace("data:image/png;base64,", "")
    elif image_data.startswith("data:image/webp;base64,"):
        image_data = image_data.replace("data:image/webp;base64,", "")
    image_data = base64.b64decode(image_data)

    # convert to jpg and crop to square
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(image_data)).convert("RGB")
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    image = image.crop((left, top, right, bottom))
    image_path = "temp_image.jpg"
    image.save(image_path, "JPEG")

    if image_data:
        result = predict("temp_image.jpg")
        os.remove(image_path)
    else:
        result = {"error": "No image data provided"}

    response = jsonify({"count": result})
    # response.headers.add("Access-Control-Allow-Origin", "*")

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
