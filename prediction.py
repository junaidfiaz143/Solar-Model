from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import argparse

ap = argparse.ArgumentParser(description="SolarPanel CNN : UTILS")
ap.add_argument("-i", "--input", required=True, type=str, default="no", help="required input image")
args = vars(ap.parse_args())

# load model
model = load_model("model/solar_model.h5")

os.system("cls")

input_name = args["input"]

# load input image
input_image = cv2.imread(input_name)
# resize input image on which model is trained
input_image = cv2.resize(input_image, (224, 224))

temp_input_image = input_image

input_image = np.expand_dims(input_image, axis=0)

print("+---------------------------+")
print("SOME LOGS")
print("+---------------------------+")
print("Input Image Name:", input_name)
print("Input Image Dimension: ", input_image.shape)

# get predictions from model
predictions = model.predict(input_image)[0]

print("{'BROKEN': 0, 'NOT-BROKEN': 1}")
print(predictions)
print(round(predictions[0]))
print("+---------------------------+")

label = round(predictions[0])
temp_input_image = cv2.resize(temp_input_image, (422, 422))

if label == 0.0:
	cv2.putText(temp_input_image, "BROKEN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
else:
	cv2.putText(temp_input_image, "NOT-BROKEN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

cv2.imwrite("outputs/predict_"+args["input"].split("/")[1], temp_input_image)
cv2.imshow("PREDICTION", temp_input_image)

cv2.waitKey(0)
cv2.destroyAllWindows()