import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import keyboard
import time

def press_key(key):
    keyboard.press(key)
    keyboard.release(key)

#need to ensure key isnt pressed every loop or get backed upkey presses
loop_count = 0
press_interval = 0


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
    landmarks = []  # Initialize landmarks list
    
    if op.multi_hand_landmarks:
        for handslms in op.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
                
        draw.draw_landmarks(frm, handslms, hands.HAND_CONNECTIONS)       
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        class_name=labels[classID].capitalize()

    
    # Output x-coordinate of the central landmark
    central_lm_x = landmarks[0][0] if landmarks else -1  # Get x-coordinate of the central landmark
    text = f"X-coordinate: {central_lm_x}"  # Format the text
    cv2.putText(frm, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2)

    cv2.putText(frm, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2)
    cv2.imshow("output", frm)

    #increment loop count
    loop_count += 1

    if loop_count >= press_interval:
        loop_count = 0  # Reset loop count

        if central_lm_x > 600:
            press_key('d')
        if central_lm_x < 400 and central_lm_x > -1:
            press_key('a')
        
        # Handle different hand gestures
        if class_name == 'Stop':
            press_key('w')
        elif class_name == 'Fist':
            press_key('s')

    
    
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

