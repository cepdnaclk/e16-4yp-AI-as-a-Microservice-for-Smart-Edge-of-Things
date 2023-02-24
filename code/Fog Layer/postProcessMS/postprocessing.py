from flask import Flask, request
app = Flask(__name__)


@app.route('/postprocess', methods=['POST'])
def postprocess_yolov5_output():
    # Get the YOLOv4 output from the request
    yolo_output = request.get_json()
    
    # Extract the bounding box coordinates and class probabilities from the YOLOv4 output
    bounding_boxes, class_probs = extract_bounding_boxes_and_class_probs(yolo_output)
    
    # Filter out bounding boxes with low confidence scores
    filtered_bounding_boxes = filter_bounding_boxes_by_confidence(bounding_boxes, class_probs, threshold=0.5)
    
    # call notification service if needed

    # call face recognition service with imagedata 
   


def extract_bounding_boxes_and_class_probs():
    return 
    #exracting bounding boxes data

def filter_bounding_boxes_by_confidence():
    return