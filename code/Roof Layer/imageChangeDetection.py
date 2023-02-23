import cv2
import os
from flask import Flask, render_template, request, redirect, send_file, url_for

app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')

os.makedirs(uploads_dir, exist_ok=True)


@app.route("/detect", methods=['POST'])
def detectChange():
# Load the first frame
    # first_frame = cv2.imread('image_stream_frame_1.jpg')
    if not request.method == "POST":
        return
    next_frame = request.files['img']

     # Load the next frame
    first_frame = cv2.imread(os.path.join(uploads_dir, "firstFrame"))
    # path
  
       # Convert the first frame to grayscale
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        # Convert the next frame to grayscale
    gray_next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        # save nextframe as firstframe
    next_frame.save(os.path.join(uploads_dir, "firstFrame"))
        # Calculate the absolute difference between the first frame and next frame
    frame_diff = cv2.absdiff(gray_first_frame, gray_next_frame)
        
        # Apply threshold to the image to make the differences more pronounced
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

        #Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If the contours have any points, this means changes have been detected
    if contours:
        #Call fog layer gayeway end point with with next frame 
        with open(os.path.join(uploads_dir, "firstFrame"), 'rb') as image_file:
            files = {'image': next_frame}
            response = request.post("url to fog gateway", files=files)
            print(response.json())           
    else:
       return False
        
        
      
    