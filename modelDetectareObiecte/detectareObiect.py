import cv2
import numpy as np

# incarcare model MobileNet SSD pre-trained
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# initializare camera
cap = cv2.VideoCapture(0)

while True:
    # captura cadru cu cadru
    ret, frame = cap.read()

    # procesare imagine pentru detectarea obiectelor
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # setare intrare in retea neuronala
    net.setInput(blob)
    detections = net.forward()

    # desenare casete pentru delimitare
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2: 
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # afisare cadru rezultat
    cv2.imshow('Object Detection', frame)

    # incheie bucla daca 'q' este apasat
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# eliberare camera si inchide toate ferestrele
cap.release()
cv2.destroyAllWindows()