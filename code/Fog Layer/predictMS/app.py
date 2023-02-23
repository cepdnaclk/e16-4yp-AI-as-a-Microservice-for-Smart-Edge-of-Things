import yolov5
from flask import Flask,jsonify,request
# load pretrained model
model = yolov5.load('yolov5s.pt')
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io

# load custom model
# //model = yolov5.load('train/best.pt')
  
app = Flask(__name__)

uploads_dir = os.path.join(app.instance_path, 'uploads')

os.makedirs(uploads_dir, exist_ok=True)

@app.route("/interference",methods=['POST'])
def hello_worlde():
    print("logs")
    dd=  {
        "name": "John Smith",
        "age": 30,
        "city": "New York"
     }
    print("dfsdfsdf")
    if  not request.method == "POST":
        return dd

    binary_image = request.files['video'].read()
    image = Image.open(io.BytesIO(binary_image))

    # video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # perform inference
    results = model(image)

    # # inference with larger input size
    # results = model(img, size=1280)

    # # inference with test time augmentation
    # results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    # print(categories)
    df = results.pandas().xyxy[0]
    labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    send = df.to_json(orient='records');
    print(df.to_json(orient='records'))
    # print(cord_thres)
    return jsonify(send)
