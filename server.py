# import the necessary packages
from imutils import build_montages
from datetime import datetime
import numpy as np
from imagezmq import imagezmq
import argparse
import imutils
import cv2
import requests

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-mW", "--montageW", required=True, type=int,
    help="montage frame width")
ap.add_argument("-mH", "--montageH", required=True, type=int,
    help="montage frame height")
args = vars(ap.parse_args())

# define API host/port
PREDICT_API_URL = 'http://localhost:8000/api/predict'
SAVE_DETECTIONS_API_URL = 'http://localhost:8000/api/detections'

# initialize the ImageHub object
imageHub = imagezmq.ImageHub(open_port='tcp://*:5555')

# initialize the consider set (class labels we care about and want
# to count), the object count dictionary, and the frame  dictionary
CONSIDER = set(["dog", "person", "car", "cat", "bird", "bicycle", "motorbike"])
prev_obj_count, cur_obj_count  = {obj: 0 for obj in CONSIDER}, {obj: 0 for obj in CONSIDER}
frameDict = {}
last_active = {}
last_check = datetime.now()
CHANGE_DETECTION_INTERVAL_SECONDS = 5

# define colors for the bounding boxes
COLORS = [tuple(255 * np.random.rand(3)) for i in range(5)]

# assign montage width and height so we can view all incoming frames
# in a single "dashboard"
mW = args["montageW"]
mH = args["montageH"]
print("[INFO] detecting: {}...".format(", ".join(obj for obj in
    CONSIDER)))

# start looping over all the frames
while True:
    # receive RPi name and frame from the RPi and acknowledge
    # the receipt
    (rpi_name, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')

    # if a device is not in the last active dictionary then it means
    # that its a newly connected device
    if rpi_name not in last_active.keys():
        print("[INFO] device started sending data: {}...".format(rpi_name))
 
    # record the last active time for the device from which we just
    # received a frame
    last_active[rpi_name] = datetime.now()

    # resize the frame to have a maximum width of 400 pixels, then
    # grab the frame dimensions and construct a blob
    #frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]

    # query API for the predictions and draw boxes if 
    # objects have been found
    preds = []
    try:
        r = requests.post(PREDICT_API_URL, json={"frame": frame.tolist()})
        resp_json = r.json()
        preds = resp_json['results']
    except:
        print("[ERROR] Unable to retrieve predictions from url: {}".format(PREDICT_API_URL))

    # draw boxes if predictions have been returned
    # do it only for the tracked objects
    cur_obj_count = {obj: 0 for obj in CONSIDER}
    if len(preds):
        for color, result in zip(COLORS, preds):
            start_y = result['topleft']['y']
            tl = (result['topleft']['x'], start_y - 15 if start_y - 15 > 15 else start_y + 15)
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            if label in CONSIDER:
                frame = cv2.rectangle(frame, tl, br, color, 2)
                frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
                cur_obj_count[label] += 1

    # check if change in detections occurred every few seconds
    if (datetime.now() - last_check).seconds > CHANGE_DETECTION_INTERVAL_SECONDS:
        last_check = datetime.now()
        if cur_obj_count != prev_obj_count:
            # extract detected objects
            obj_detected = [o[0] for o in cur_obj_count.items() if o[1] > 0] 
            try:
                requests.post(SAVE_DETECTIONS_API_URL, json={"frame": frame.tolist(), "obj_detected": obj_detected})
            except:
                print("[ERROR] Unable to save detections at url: {}".format(SAVE_DETECTIONS_API_URL))

            prev_obj_count = cur_obj_count
            
    # draw the object count on the frame
    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in cur_obj_count.items())
    cv2.putText(frame, label, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.358, (0, 255,0), 1)

    # draw the sending device name on the frame
    cv2.putText(frame, rpi_name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 255), 1)

    # update the new frame in the frame dictionary
    frameDict[rpi_name] = frame
 
    # build a montage using images in the frame dictionary
    montages = build_montages(frameDict.values(), (w, h), (mW, mH))
 
    # display the montage(s) on the screen
    for (i, montage) in enumerate(montages):
        cv2.imshow("Home monitor ({})".format(i), montage)
 
    # detect any kepresses
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
cv2.destroyAllWindows()
