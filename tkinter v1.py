import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pynput.keyboard import Key, Controller

#class for app
class App:
    def __init__(self, window, window_title):
        #window
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="white")

        #for keypresses
        self.keyboard = Controller()

        # Load the gesture recognition model
        self.model = load_model('gestures')
        
        #OpenCV video capture
        self.vid = cv2.VideoCapture(0)

        #title label
        self.program_title = tk.Label(self.window, text = "\nHand Gesture Control System:", font=("Arial", 18, "bold underline"),bg = "white")
        self.program_title.pack(anchor=tk.N)
        
        #buffer labels
        self.top_buffer_label = tk.Label(self.window, text="", font=("Arial", 12),bg = "white")
        self.bottom_buffer_label = tk.Label(self.window, text="", font=("Arial", 12),bg = "white")

        # Create a frame widget as a container for the canvas
        canvas_frame = tk.Frame(window, bg = "#000000", bd=2)
        canvas_frame.pack(padx=10,pady=10)
        
        #Create a canvas that can fit the video source with buffers
        self.canvas = tk.Canvas(canvas_frame, bg = "white", width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.top_buffer_label.pack(anchor=tk.N)
        self.canvas.pack(anchor=tk.CENTER)
        self.bottom_buffer_label.pack(anchor=tk.N)
        
        #Button to start/stop the video
        self.btn_start_stop = tk.Button(window, text="Start", width=10, command=self.start_stop_video)
        self.btn_start_stop.pack(anchor=tk.CENTER, expand=True)

        #indicates if video is playing
        self.is_playing = False
        #Initialize photo attribute
        self.photo = None

        # Mediapipe Hands initialization
        self.hands = mp.solutions.hands
        self.hands_mesh = self.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.draw = mp.solutions.drawing_utils

        # Load gesture labels
        f = open('gesture.names.txt','r')
        self.labels = f.read().split('\n')
        f.close()

        #list to store hand labels and gestures
        self.hand_labels = []

        # Create label widgets for right and left hand
        self.hands_title_label = tk.Label(window, text="", font=("Arial", 12, "bold underline"),bg = "white")
        self.hands_title_label.pack(anchor =tk.W)
        self.right_hand_label = tk.Label(self.window, text="", font=("Arial", 12),bg = "white")
        self.right_hand_label.pack(anchor=tk.W)
        self.left_hand_label = tk.Label(self.window, text="", font=("Arial", 12),bg = "white")
        self.left_hand_label.pack(anchor=tk.W)
        self.right_hand_xcoord_label = tk.Label(self.window, text="", font=("Arial", 12),bg = "white")
        self.right_hand_xcoord_label.pack(anchor=tk.W)
        self.right_hand_ycoord_label = tk.Label(self.window, text="", font=("Arial", 12),bg = "white")
        self.right_hand_ycoord_label.pack(anchor=tk.W)
        self.hand_buffer_label = tk.Label(window, text="", font=("Arial", 12),bg = "white")
        self.hand_buffer_label.pack(anchor=tk.W)
        
        #run mainloop
        self.update()
        self.window.mainloop()

    #for starting and stopping video capture
    def start_stop_video(self):
        if self.is_playing:
            self.btn_start_stop.config(text="Start")
        else:
            self.btn_start_stop.config(text="Stop")
        self.is_playing = not self.is_playing

    #recognizes gesture 
    def gesture_recognition(self, hand_label,landmarks):
        prediction = self.model.predict([landmarks])
        classID = np.argmax(prediction)
        gesture_name = self.labels[classID].capitalize()
        # Append hand label and gesture to the list
        self.hand_labels.append((hand_label, gesture_name))
        return gesture_name

    #presses passed key on keyboard 
    def press_key(self,key):
        self.keyboard.press(key)
        self.keyboard.release(key)

    #update self
    def update(self):
        if self.is_playing:
            # Get a frame from the video source
            ret, frm = self.vid.read()
            
            if ret:
                #update labels for left and right hand
                self.hands_title_label.config(text = "Hand Tracking Information:")
                self.right_hand_label.config(text="Right Hand:")
                self.left_hand_label.config(text="Left Hand:")
                self.right_hand_xcoord_label.config(text = "Right Hand X Coordinates:")
                self.right_hand_ycoord_label.config(text = "Right Hand Y Coordinates:")
                
                #get frame dimensions
                y, x, c = frm.shape

                # Flip the frame
                frm = cv2.flip(frm, 1)
            
                rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

                # Calculate the x-coordinate one-third of the way across the screen
                line_x = int(x / 3)

                # Draw a vertical line at the calculated x-coordinate
                cv2.line(frm, (line_x, 0), (line_x, y), (0, 255, 0), 2)

                # Square for deadzone of controls
                square_x = int(x / 3) * 2
                square_size = int(x/7)

                # Calculate the coordinates of the square's vertices
                top_left = (square_x - square_size // 2, y // 2 - square_size // 2)
                bottom_right = (square_x + square_size // 2, y // 2 + square_size // 2)

                # Draw the square
                cv2.rectangle(frm, top_left, bottom_right, (0, 0, 255), 2)

                # Process hand landmarks with Mediapipe Hands
                results = self.hands_mesh.process(rgb)

                # Clear hand-related variables
                self.hand_labels.clear()
                right_hand_landmarks = None
                left_hand_landmarks = None
                right_gesture_name = None
                left_gesture_name = None
                
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        
                        self.draw.draw_landmarks(frm, hand_landmarks, self.hands.HAND_CONNECTIONS,
                                                self.draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                                self.draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                            
                        landmarks = []  # Initialize landmarks list
                        for lm in hand_landmarks.landmark:
                            # Center of the screen is 0,0
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)
                            landmarks.append([lmx, lmy])

                        # Check if hand is right and on the right side of the line
                        if hand_label == 'Right' and landmarks[self.hands.HandLandmark.MIDDLE_FINGER_MCP.value][0] > line_x:
                            right_hand_landmarks = landmarks
                            right_gesture_name = self.gesture_recognition(hand_label,right_hand_landmarks)

                        # Check if hand is left and on the left side of the line
                        if hand_label == 'Left' and landmarks[self.hands.HandLandmark.MIDDLE_FINGER_MCP.value][0] < line_x:
                            left_hand_landmarks = landmarks
                            left_gesture_name = self.gesture_recognition(hand_label,left_hand_landmarks)
                
                # Convert the frame to RGB format
                frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

                # Convert the frame to ImageTk format
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frm_rgb))
                
                # Display the frame in the Tkinter window 
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                #Update labels with hand gesture info
                if right_gesture_name:
                    self.right_hand_label.config(text=f"Right Hand: {right_gesture_name}")
                if left_gesture_name:
                    self.left_hand_label.config(text=f"Left Hand: {left_gesture_name}")

                # Output x/y-coordinate of the central landmark on the right hand 
                central_lm_x = right_hand_landmarks[9][0] - int(x/3) if right_hand_landmarks else None  # Get x-coordinate of the central landmark
                central_lm_y = right_hand_landmarks[9][1] if right_hand_landmarks else None  # Get y-coordinate of the central landmark

                #want right hand x coordinates centred on second two thirds of screen 
                if central_lm_x and central_lm_y:
                    x_coord_changer = int(x/3) 
                    central_lm_x = central_lm_x - x_coord_changer
                    y_coord_changer = int(y/2)
                    central_lm_y = (central_lm_y - y_coord_changer) * -1

                    #make coords equal 0 if inside the square
                    if central_lm_x < 125 and  central_lm_x > - 125 and central_lm_y < 125 and  central_lm_y > -125:
                        central_lm_x = 0
                        central_lm_y = 0

                    #update coords labels
                    self.right_hand_xcoord_label.config(text = f"Right Hand X Coordinates: {central_lm_x}")
                    self.right_hand_ycoord_label.config(text = f"Right Hand Y Coordinates: {central_lm_y}")

                    #button presses on hand position 
                    if central_lm_x > square_size/2:
                       self.press_key('l')
                    if central_lm_x < -square_size/2:
                       self.press_key('j')
                    if central_lm_y > square_size/2:
                       self.press_key('i')
                    if central_lm_y < -square_size/2:
                       self.press_key(',')
        

                #button presses on gestures 
                if right_gesture_name == 'Fist':
                    self.press_key('q')

                if right_gesture_name == 'Call me':
                    self.press_key('w')

                if left_gesture_name == 'Fist':
                    self.press_key('e')

                if left_gesture_name == 'Call me':
                    self.press_key('r')
     
        else:
            # Clear canvas if video is not playing
            self.canvas.delete("all")

            #remove labels for hands
            self.hands_title_label.config(text = "")
            self.right_hand_label.config(text="")
            self.left_hand_label.config(text="")
            self.right_hand_xcoord_label.config(text = "")
            self.right_hand_ycoord_label.config(text = "")
        
        # Call the update function after 10 milliseconds
        self.window.after(10, self.update)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the App class
App(tk.Tk(), "Hand Gesture Control Sytem")
