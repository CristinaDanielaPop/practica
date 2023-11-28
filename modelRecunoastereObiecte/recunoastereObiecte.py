import cv2
import numpy as np

# capteaza imagine cu camera Raspberry Pi
image = cv2.imread('path_to_image.jpg')
image = cv2.resize(image, (32, 32))

# normalizare imagine
image = image / 255.0
image = np.expand_dims(image, axis=0)

# recunoasterea obiectelor
predictions = model.predict(image)

# afisare rezultate
class_index = np.argmax(predictions)
print(f'Clasa recunoscută: {class_index}')