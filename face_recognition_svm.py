# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

import face_recognition
from sklearn import svm
import os
import cv2
import pickle


# Get pk File
with open('Dumbed_Model/Trained_Model/MAIN/omar_Other.pk','rb') as f:
    mypickle = pickle.load(f)

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('SVC_Testimgs/TeamPhotos/313029380_820622142513447_5367526483230096703_n.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)


# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    
    propa_name = mypickle.predict_proba([test_image_enc])
    # Getting Model Accuracy
    max_acc = propa_name.max()*100
    if (max_acc) > 90:
        name = mypickle.predict([test_image_enc]) 
        print(*name)
        print("The Model Accuracy Is = " , round(max_acc,2) , "%")
    else:
        name=['UnKnown']
        print(*name)
        print("The Model Accuracy Is = " , round(max_acc,2) , "%")

# Print Accuracy
print("propa_name",propa_name)

    # Show Result On Screen
while True:
    ResImg = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    for face in face_locations:
        top, right, bottom, left = face
        cv2.rectangle(ResImg, (left, top), (right, bottom), (255, 0, 0), 2)
        # cv2.rectangle(ResImg, (left, top+109), (right, bottom+30), (0, 255, 255),-1) # make a background for the name
        cv2.putText(ResImg, *name , (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(ResImg, "accuracy : "+ str(round(max_acc,2))+'%' , (left, bottom + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow('img',ResImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
