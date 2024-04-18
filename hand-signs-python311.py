import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model


model = load_model('gestures')

hands = mp.solutions.hands
hands_mesh = hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

f = open('gesture.names.txt','r')
labels = f.read().split('\n')
f.close()
print(labels)

cap = cv2.VideoCapture(0)

while True:
    _, frm = cap.read()
    x , y, c = frm.shape  
    
    rgb = cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
    
    op = hands_mesh.process(rgb)

    class_name=''
    
    if op.multi_hand_landmarks:
        landmarks = []
        for handslms in op.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
                
        draw.draw_landmarks(frm, handslms, hands.HAND_CONNECTIONS)       
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        class_name=labels[classID].capitalize()
        

    cv2.putText(frm,class_name,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225),2)
    cv2.imshow("output",frm)
    
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
