import cv2
from random import randint


dnn = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


with open('classes.txt') as f:
    classes = f.read().strip().splitlines()



video_paths = {
    '1': '../Video.mp4',
    '2': '../v.mp4'
}


print("Press 1 for first video and 2 for second video")


key1 = input()


video_file_path = video_paths[key1]
capture = cv2.VideoCapture(video_file_path)


capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

color_map = {}

while True:
 
    ret, frame = capture.read()

    if not ret:
       
        break

    frame = cv2.flip(frame, 1)

    
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1)

   
    if key == 27:  # Esc key to exit
        break

    elif key == 13:  # Enter key to reset colors
        color_map = {}


capture.release()
cv2.destroyAllWindows()
