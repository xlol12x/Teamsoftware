import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import vgamepad as vg
#emulating an xbox 360 controller 
gamepad = vg.VX360Gamepad()

global labels
global hand_labels

x_button_pressed = False
y_button_pressed = False
a_button_pressed = False 

model = load_model('gestures')

hands = mp.solutions.hands
hands_mesh = hands.Hands(max_num_hands=2,min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

f = open('gesture.names.txt','r')
labels = f.read().split('\n')
f.close()
print(labels)

#use 1080p resolution and 24fps
cap = cv2.VideoCapture(0)

def gesture_recognition(landmarks):
    prediction = model.predict([landmarks])
    classID = np.argmax(prediction)
    gesture_name = labels[classID].capitalize()
    # Append hand label and gesture to the list
    hand_labels.append((hand_label, gesture_name))
    return gesture_name


#list to store hand labels and gestures
hand_labels = []
right_hand_landmarks = None
while True:
    _, frm = cap.read()
    y, x, c = frm.shape  

    frm = cv2.resize(frm, (1280, 720))
    
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    #Calculate the x-coordinate one-third of the way across the screen
    line_x = int(x / 3)

    #Draw a vertical line at the calculated x-coordinate
    cv2.line(frm, (line_x, 0), (line_x, y), (0, 255, 0), 2)

    #square for deadzone of controls
    square_x = int(x/3) * 2
    square_size = 150

    #Calculates the coordinates of the square's vertices
    top_left = (square_x - square_size // 2, y // 2 - square_size // 2)
    bottom_right = (square_x + square_size // 2, y // 2 + square_size // 2)

    #draw ssquare
    cv2.rectangle(frm, top_left, bottom_right, (0, 0, 255), 2)

    results = hands_mesh.process(rgb)

    #clear each loop 
    hand_labels.clear()
    right_hand_landmarks = None
    left_hand_landmarks = None
    right_gesture_name = None
    left_gesture_name = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            
            draw.draw_landmarks(frm, hand_landmarks, hands.HAND_CONNECTIONS,
                                draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                
            landmarks = []  # Initialize landmarks list
            for lm in hand_landmarks.landmark:
                # Center of the screen is 0,0
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            #check if hand is right and right side of line 
            if hand_label == 'Right'and landmarks[hands.HandLandmark.MIDDLE_FINGER_MCP.value][0] > line_x:
                right_hand_landmarks = landmarks
                right_gesture_name = gesture_recognition(right_hand_landmarks)

            #check if hand is left and left side of the line
            if hand_label == 'Left' and landmarks[hands.HandLandmark.MIDDLE_FINGER_MCP.value][0] < line_x:
                left_hand_landmarks = landmarks
                left_gesture_name = gesture_recognition(left_hand_landmarks)

    # Draw labels 
    for i, (hand_label, gesture_name) in enumerate(hand_labels):
        label_text = f"Hand {i + 1}: {hand_label}, Gesture: {gesture_name}"
        cv2.putText(frm, label_text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Output x/y-coordinate of the central landmark
    central_lm_x = right_hand_landmarks[9][0] - int(x/3) if right_hand_landmarks else 0  # Get x-coordinate of the central landmark
    central_lm_y = right_hand_landmarks[9][1] if right_hand_landmarks else 0  # Get y-coordinate of the central landmark

    #want right hand x coordinates centred on second two thirds of screen 
    if central_lm_x:
        x_coord_changer = int(x/3) 
        central_lm_x = central_lm_x - x_coord_changer
        y_coord_changer = int(y/2)
        central_lm_y = (central_lm_y - y_coord_changer) * -1

    #make coords equal 0 if inside the square
    if central_lm_x < 75 and  central_lm_x > - 75 and central_lm_y < 75 and  central_lm_y > -75:
        central_lm_x = 0
        central_lm_y = 0

    #output right hand coords 
    text = f"Right Hand X-coordinate: {central_lm_x}, Y-coordinate: {central_lm_y}"  # Format the text
    cv2.putText(frm, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2)
            
    cv2.imshow("Hand Tracking", frm)

    #button presses
    if right_gesture_name == 'Fist' or right_gesture_name == 'Rock' and not x_button_pressed:
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        x_button_pressed = True


    elif right_gesture_name != 'Fist' or right_gesture_name == 'Rock' and x_button_pressed:  # Release the button for other gestures
        gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        x_button_pressed = False

    if left_gesture_name == 'Fist' and not y_button_pressed:
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        y_button_pressed = True

    elif left_gesture_name != 'Fist'and y_button_pressed:  # Release the button for other gestures
        gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        y_button_pressed = False

    if left_gesture_name == 'Call me' and not a_button_pressed:
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        a_button_pressed = True

    elif left_gesture_name != 'Call me' and a_button_pressed:
        gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        a_button_pressed = False
    
    #scale coordinates to control stick 
    gamepad_x = central_lm_x * 90
    gamepad_y = central_lm_y * 110

    if gamepad_x > 32767:
        gamepad_x = 32767
    elif gamepad_x < -32768:
        gamepad_x = -32768

    if gamepad_y > 32767:
        gamepad_y = 32767
    elif gamepad_y < -32768:
        gamepad_y = -32768
    
    gamepad.left_joystick(x_value= gamepad_x, y_value= gamepad_y)  # values between -32768 and 32767
    gamepad.update()

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
