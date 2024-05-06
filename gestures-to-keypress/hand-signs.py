import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pynput.keyboard import Key, Controller
import vgamepad as vg


#class for app
class App:
    def __init__(self, window, window_title):
        #window
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="white")
        self.window.resizable(False, False)

        #for keypresses
        self.keyboard = Controller()

        #for joystick controls
        #emulating an xbox 360 controller 
        self.gamepad = vg.VX360Gamepad()

        #indicates if a button is pressed (for controller)
        self.x_button_pressed = False
        self.y_button_pressed = False
        self.a_button_pressed = False
        self.b_button_pressed = False

        # Load the gesture recognition model
        self.model = load_model('gestures')
        
        #OpenCV video capture
        self.vid = cv2.VideoCapture(0)

        #title label
        self.program_title = tk.Label(self.window, text = "Hand Gesture Control System:", font=("Arial", 24, "bold underline"),bg = "white")
        self.program_title.pack(anchor=tk.N, pady=20)

        #Create a frame widget as a container for the instructions/config
        detail_frame = tk.Frame(window, bg = "#000000", bd=2)
        detail_frame.pack(side = "left",padx=10,pady=10, anchor = tk.N)
        
        #canvas for instructions
        self.instruction_canvas = tk.Canvas(detail_frame, bg = "white", width=400, height=350, highlightthickness=0)
        self.instruction_canvas.pack()
        self.instruction_canvas.pack_propagate(0)

        #frame for config
        self.config_frame = tk.Frame(detail_frame, bg="white",highlightthickness=0)
        self.config_frame.pack()

        #config of notebook
        #Configure the style
        style = ttk.Style()
        #Set tabs position to north (centered)
        style.configure("Custom.TNotebook", tabposition="n",background="white")  
        # Set tab padding and background color for active tab
        style.configure("Custom.TNotebook.Tab", background="#808080", padding=[65, 10])

    
        #Notebook for config tabs
        self.notebook = ttk.Notebook(self.config_frame, style="Custom.TNotebook")
        self.notebook.pack()

        #Canvas for config tab 1
        self.config_canvas1 = tk.Canvas(self.notebook, bg="white", width=395, height=430, highlightthickness=0)
        self.config_canvas1.pack()
        self.config_canvas1.pack_propagate(0)
        self.notebook.add(self.config_canvas1, text="Keyboard")

        #Canvas for config tab 2
        self.config_canvas2 = tk.Canvas(self.notebook, bg="white", width=395, height=430, highlightthickness=0)
        self.config_canvas2.pack()
        self.config_canvas2.pack_propagate(0)
        self.notebook.add(self.config_canvas2, text="Controller")

        # Draw a line between the two canvases
        self.draw_separator()

        #insert instructions
        self.create_instructions()

        #inputs for keyboard config
        self.keyboard_config()

        #labels for controller config
        self.controller_config()
        
        #buffer label
        self.top_buffer_label = tk.Label(self.window, text="", font=("Arial", 12),bg = "white")

        # Create a frame widget as a container for the canvas
        canvas_frame = tk.Frame(window, bg = "#000000", bd=2)
        canvas_frame.pack(padx=20,pady=10)
        
        #Create a canvas that can fit the video source with buffers
        self.canvas = tk.Canvas(canvas_frame, bg = "light blue", width=960, height=540,highlightthickness=0)
        self.top_buffer_label.pack(anchor=tk.N)
        self.canvas.pack(anchor=tk.CENTER)
    
        #Button to start/stop the video
        style = ttk.Style()
        style.configure('SquareButton.TButton', foreground="black", background="white",
                        borderwidth=5, relief="flat",padding=15, width=15, font=('Arial', 14), borderstyle="solid")

        self.btn_start_stop = ttk.Button(window, text="Start", style='SquareButton.TButton', command=self.start_stop_video)
        self.btn_start_stop.pack(anchor=tk.CENTER, expand=True)
        
        #indicates if video is playing
        self.is_playing = False
        #Initialize photo attribute
        self.photo = None

        # Mediapipe Hands initializationr
        self.hands = mp.solutions.hands
        self.hands_mesh = self.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.draw = mp.solutions.drawing_utils

        # Load gesture labels
        f = open('gesture.names.txt','r')
        self.labels = f.read().split('\n')
        f.close()

        #list to store hand labels and gestures
        self.hand_labels = []

        #canvas for organizing the labels
        tracking_canvas = tk.Canvas(window, bg="white", highlightthickness=0, width=window.winfo_width() - 20)
        tracking_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        #left frame for left-hand labels
        left_canvas = tk.Canvas(tracking_canvas, bg="white",highlightthickness=0)
        left_canvas.pack(side="left", fill="both", expand=True)
        left_canvas.pack_propagate(0)

        #right frame for right-hand labels
        right_canvas = tk.Canvas(tracking_canvas, bg="white",highlightthickness=0)
        right_canvas.pack(side="right", fill="both", expand=True)
        right_canvas.pack_propagate(0)

        # Create label widgets for right and left hand
        self.hands_title_label = tk.Label(left_canvas, text="Hand Tracking Information:", font=("Arial", 12, "bold underline"), bg="white")
        self.hands_title_label.pack(anchor="w", padx=7)
        self.right_hand_label = tk.Label(left_canvas, text="", font=("Arial", 12), bg="white")
        self.right_hand_label.pack(anchor="w", padx=7)
        self.left_hand_label = tk.Label(left_canvas, text="", font=("Arial", 12), bg="white")
        self.left_hand_label.pack(anchor="w", padx=7)
        self.right_hand_xcoord_label = tk.Label(left_canvas, text="", font=("Arial", 12), bg="white")
        self.right_hand_xcoord_label.pack(anchor="w", padx=7)
        self.right_hand_ycoord_label = tk.Label(left_canvas, text="", font=("Arial", 12), bg="white")
        self.right_hand_ycoord_label.pack(anchor="w", padx=7)
        self.hand_buffer_label = tk.Label(left_canvas, text="", font=("Arial", 12), bg="white")
        self.hand_buffer_label.pack(anchor="w", padx=7)

        # Create label widgets for current output
        self.outputs_title_label = tk.Label(right_canvas, text="Current Output:", font=("Arial", 12, "bold underline"), bg="white")
        self.outputs_title_label.pack(anchor="w", padx=(250,0))
        self.right_hand_output_label = tk.Label(right_canvas, text="", font=("Arial", 12), bg="white")
        self.right_hand_output_label.pack(anchor="w", padx=(250,0))
        self.left_hand_output_label = tk.Label(right_canvas, text="", font=("Arial", 12), bg="white")
        self.left_hand_output_label.pack(anchor="w", padx=(250,0))
        self.vertical_output_label = tk.Label(right_canvas, text="", font=("Arial", 12), bg="white")
        self.vertical_output_label.pack(anchor="w", padx=(250,0))
        self.horizontal_hand_output_label = tk.Label(right_canvas, text="", font=("Arial", 12), bg="white")
        self.horizontal_hand_output_label.pack(anchor="w", padx=(250,0))

        #used for storing the keyboard inputs the user wants
        self.forward_key = ""
        self.backward_key = ""
        self.left_key = ""
        self.right_key = ""
        self.left_fist_key = ""
        self.left_call_key = ""
        self.right_fist_key = ""
        self.right_call_key = ""
        
        #run mainloop
        self.update()
        self.window.mainloop()

    #for starting and stopping video capture
    def start_stop_video(self):
        if self.is_playing:
            self.btn_start_stop.config(text="Start")
        else:
            self.btn_start_stop.config(text="Stop")
            #get desired input values
            self.forward_key = self.forward_entry.get()
            self.backward_key = self.backward_entry.get()
            self.left_key = self.left_entry.get()
            self.right_key = self.right_entry.get()
            self.left_fist_key = self.left_fist_entry.get()
            self.left_call_key = self.left_call_entry.get()
            self.right_fist_key = self.right_fist_entry.get()
            self.right_call_key = self.right_call_entry.get()
            
        self.is_playing = not self.is_playing

    #recognizes gesture 
    def gesture_recognition(self, hand_label,landmarks):
        prediction = self.model.predict([landmarks])
        classID = np.argmax(prediction)
        gesture_name = self.labels[classID].capitalize()
        # Append hand label and gesture to the list
        self.hand_labels.append((hand_label, gesture_name))
        return gesture_name

    #draw line between instruction and confif
    def draw_separator(self):
        self.instruction_canvas.create_line(0, 300, 400, 300, fill="black", width=3)

    #presses passed key on keyboard 
    def press_key(self,key):
        self.keyboard.press(key)
        self.keyboard.release(key)

    #creates instructions
    def create_instructions(self):
        instruction_title_label = tk.Label(self.instruction_canvas, text="Instructions", font=("Arial", 16, "bold underline"), bg="white")
        instruction_title_label.pack(anchor="w", padx=10, pady=10)

        instructions_label1 = tk.Label(self.instruction_canvas, text="Choose input method, either keyboard or controller.", font=("Arial", 12), bg="white")
        instructions_label2 = tk.Label(self.instruction_canvas, text="Select what the gestures input, or use default values.", font=("Arial", 12), bg="white")
        instructions_label3 = tk.Label(self.instruction_canvas, text="When ready, press start to begin video capture.", font=("Arial", 12), bg="white")
        instructions_label4 = tk.Label(self.instruction_canvas, text="Hands must be on correct sides of the green line.", font=("Arial", 12), bg="white")
        instructions_label5 = tk.Label(self.instruction_canvas, text="Left hand is used purely for gesture based inputs.", font=("Arial", 12), bg="white")
        instructions_label6 = tk.Label(self.instruction_canvas, text="Right hand position controls directional controls.", font=("Arial", 12), bg="white")
        instructions_label7 = tk.Label(self.instruction_canvas, text="Right hand can also be used for gesture based inputs.", font=("Arial", 12), bg="white")
        instructions_label8 = tk.Label(self.instruction_canvas, text="Once started, click into the program you wish to control", font=("Arial", 12), bg="white")
        instructions_label1.pack(anchor="w",padx=10)
        instructions_label2.pack(anchor="w",padx=10)
        instructions_label3.pack(anchor="w",padx=10)
        instructions_label4.pack(anchor="w",padx=10)
        instructions_label5.pack(anchor="w",padx=10)
        instructions_label6.pack(anchor="w",padx=10)
        instructions_label7.pack(anchor="w",padx=10)
        instructions_label8.pack(anchor="w",padx=10)

        Config_title_label = tk.Label(self.instruction_canvas, text="Configuration", font=("Arial", 16, "bold underline"), bg="white")
        Config_title_label.pack(anchor="w", padx=10, pady=(70,0))

    #for the keyboard config
    def keyboard_config(self):
        #Create input labels 
        forward_label, self.forward_entry = self.create_input(self.config_canvas1, "Forward Key:","i")
        backward_label, self.backward_entry = self.create_input(self.config_canvas1, "Backward Key:",",")
        left_label, self.left_entry = self.create_input(self.config_canvas1, "Left Key:","j")
        right_label, self.right_entry = self.create_input(self.config_canvas1, "Right Key:","l")
        left_fist_label, self.left_fist_entry = self.create_input(self.config_canvas1, "Left Hand Fist:","r")
        left_call_label, self.left_call_entry = self.create_input(self.config_canvas1, "Left Hand Call:","e")
        right_fist_label, self.right_fist_entry = self.create_input(self.config_canvas1, "Right Hand Fist:","q")
        right_call_label, self.right_call_entry = self.create_input(self.config_canvas1, "Right Hand Call:","w")

    #Function to create keyboard configuration inputs
    def create_input(self,canvas, label_text,default_value):
        #validates legnth of input
        def validate_input(new_value):
            if len(new_value) <= 1:
                return True
            else:
                return False

        vcmd = (self.window.register(validate_input), '%P')

        label = tk.Label(canvas, text=label_text, bg="white")
        label.pack(anchor=tk.CENTER, pady = 5)
        
        entry = tk.Entry(canvas, validate="key", validatecommand=vcmd)
        entry.insert(0, default_value)
        entry.pack(anchor=tk.CENTER)
        return label, entry

    #labels for controller config (no button customisation here just what does what is stated)
    #customisation feels redundant
    def controller_config(self):
        controller_label1 = tk.Label(self.config_canvas2, text="Joystick control based on hand position", font=("Arial", 12), bg="white")
        controller_label2 = tk.Label(self.config_canvas2, text="Right fist: X button", font=("Arial", 12), bg="white")
        controller_label3 = tk.Label(self.config_canvas2, text="Right call: B button", font=("Arial", 12), bg="white")
        controller_label4 = tk.Label(self.config_canvas2, text="Left fist: Y button", font=("Arial", 12), bg="white")
        controller_label5 = tk.Label(self.config_canvas2, text="Left call: A button", font=("Arial", 12), bg="white")

        controller_label1.pack(anchor=tk.CENTER,pady=(20,0))
        controller_label2.pack(anchor=tk.CENTER)
        controller_label3.pack(anchor=tk.CENTER)
        controller_label4.pack(anchor=tk.CENTER)
        controller_label5.pack(anchor=tk.CENTER)

    #update self
    def update(self):
        if self.is_playing:
            
            #Get a frame from the video source
            ret, frm = self.vid.read()
            #resize frame
            frm = cv2.resize(frm, (960, 540))
            
            if ret:
                #get frame dimensions
                y, x, c = frm.shape

                #Flip the frame
                frm = cv2.flip(frm, 1)
            
                rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

                #Calculate the x-coordinate one-third of the way across the screen
                line_x = int(x / 3)

                #Draw a vertical line at the calculated x-coordinate
                cv2.line(frm, (line_x, 0), (line_x, y), (0, 255, 0), 2)

                #Square for deadzone of controls
                square_x = int(x / 3) * 2
                square_size = int(x/7)

                #Calculate the coordinates of the square's vertices
                top_left = (square_x - square_size // 2, y // 2 - square_size // 2)
                bottom_right = (square_x + square_size // 2, y // 2 + square_size // 2)

                #Draw the square
                cv2.rectangle(frm, top_left, bottom_right, (0, 0, 255), 2)

                #Process hand landmarks with Mediapipe Hands
                results = self.hands_mesh.process(rgb)

                #Clear hand-related variables
                self.hand_labels.clear()
                right_hand_landmarks = None
                left_hand_landmarks = None
                right_gesture_name = None
                left_gesture_name = None

                #if landmarks exist, draw them
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        
                        self.draw.draw_landmarks(frm, hand_landmarks, self.hands.HAND_CONNECTIONS,
                                                self.draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                                self.draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                        
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            if idx == 9:  # Replace YOUR_SPECIFIC_LANDMARK_INDEX with the index of the landmark you want to draw
                                lmx = int(landmark.x * x)
                                lmy = int(landmark.y * y)
                                cv2.circle(frm, (lmx, lmy), 4, (255, 0, 0), -1)  # Draw the landmark with color (255, 0, 0) (red)
                                break  # Break the loop once the specific landmark is drawn
                        
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
                else:
                    self.right_hand_label.config(text=f"Right Hand:")
                    
                if left_gesture_name:
                    self.left_hand_label.config(text=f"Left Hand: {left_gesture_name}")
                else:
                    self.left_hand_label.config(text=f"Left Hand:")

                # Output x/y-coordinate of the central landmark on the right hand 
                central_lm_x = right_hand_landmarks[9][0] - int(x/3) if right_hand_landmarks else None  # Get x-coordinate of the central landmark
                central_lm_y = right_hand_landmarks[9][1] if right_hand_landmarks else None  # Get y-coordinate of the central landmark

                #determines control scheme 
                active_tab_index = self.notebook.select()
                active_tab_text = self.notebook.tab(active_tab_index, "text")
                #print(active_tab_text)

                #used to display the presssed buttons/keys
                horizontal_output = ""
                vertical_output = ""
                right_output = ""
                left_output = ""
                       
                #want right hand x coordinates centred on second two thirds of screen 
                if central_lm_x and central_lm_y:
                    x_coord_changer = int(x/3) 
                    central_lm_x = central_lm_x - x_coord_changer
                    y_coord_changer = int(y/2)
                    central_lm_y = (central_lm_y - y_coord_changer) * -1

                    #make coords equal 0 if inside the square
                    if central_lm_x < square_size/2 and  central_lm_x > -(square_size/2) and central_lm_y < square_size/2 and  central_lm_y > -(square_size/2):
                        central_lm_x = 0
                        central_lm_y = 0

                    #update coords labels
                    self.right_hand_xcoord_label.config(text = f"Right Hand X Coordinates: {central_lm_x}")
                    self.right_hand_ycoord_label.config(text = f"Right Hand Y Coordinates: {central_lm_y}")
                    if active_tab_text == "Keyboard":
                        #key presses on hand position 
                        if central_lm_x > square_size/2:
                            self.press_key(self.right_key)
                            horizontal_output = self.right_key
                        elif central_lm_x < -square_size/2:
                            self.press_key(self.left_key)
                            horizontal_output = self.left_key
                           
                        if central_lm_y > square_size/2:
                            self.press_key(self.forward_key)
                            vertical_output = self.forward_key
                        elif central_lm_y < -square_size/2:
                            self.press_key(self.backward_key)
                            vertical_output = self.backward_key

                    if active_tab_text == "Controller":
                        #joystick controls
                        #scale coordinates to control stick 
                        gamepad_x = central_lm_x * 130
                        gamepad_y = central_lm_y * 150

                        if gamepad_x > 32767:
                            gamepad_x = 32767
                        elif gamepad_x < -32768:
                            gamepad_x = -32768

                        if gamepad_y > 32767:
                            gamepad_y = 32767
                        elif gamepad_y < -32768:
                            gamepad_y = -32768
                    
                        self.gamepad.left_joystick(x_value= gamepad_x, y_value= gamepad_y)  # values between -32768 and 32767
                        self.gamepad.update()

                        horizontal_output = gamepad_y
                        vertical_output = gamepad_x
                      
                #button presses on gestures (keyboard and controller version
                if active_tab_text == "Keyboard":
                    if right_gesture_name == 'Fist':
                        self.press_key(self.right_fist_key)
                        right_output = self.right_fist_key
                    elif right_gesture_name == 'Call me':
                        self.press_key(self.right_call_key)
                        right_output = self.right_call_key
            
                    if left_gesture_name == 'Fist':
                        self.press_key(self.left_fist_key)
                        left_output = self.left_fist_key
                    elif left_gesture_name == 'Call me':
                        self.press_key(self.left_call_key)
                        left_output = self.left_call_key
                        
                elif active_tab_text == "Controller":
                    if right_gesture_name == 'Fist' and not self.x_button_pressed:
                        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
                        self.x_button_pressed = True
                    elif right_gesture_name != 'Fist' and self.x_button_pressed:
                        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
                        self.x_button_pressed = False

                    if right_gesture_name == 'Call me' and not self.b_button_pressed:
                        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
                        self.b_button_pressed = True
                    elif right_gesture_name != 'Call me' and self.b_button_pressed:
                        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
                        self.b_button_pressed = False

                    if left_gesture_name == 'Fist' and not self.y_button_pressed:
                        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
                        self.y_button_pressed = True
                    elif left_gesture_name != 'Fist'and self.y_button_pressed:
                        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
                        self.y_button_pressed = False
                    if left_gesture_name == 'Call me' and not self.a_button_pressed:
                        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                        self.a_button_pressed = True
                    elif left_gesture_name != 'Call me' and self.a_button_pressed:
                        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                        self.a_button_pressed = False

                    self.gamepad.update()

                    #for output
                    if self.x_button_pressed == True:
                        right_output = "X Button"
                    if self.b_button_pressed == True:
                        right_output = "B Button"
                    if self.y_button_pressed == True:
                        left_output = "Y Button"
                    if self.a_button_pressed == True:
                        left_output = "A Button"
                        
                #update output info
                self.right_hand_output_label.configure(text = f"Right Gesture Output: {right_output}")
                self.left_hand_output_label.configure(text = f"Left Gesture Output: {left_output}")
                self.vertical_output_label.configure(text = f"Vertical Position Output: {vertical_output}")
                self.horizontal_hand_output_label.configure(text = f"Horizontal Position Output: {horizontal_output}")

        else:
            # Clear canvas if video is not playing
            self.canvas.delete("all")

            #set labels back to default 
            self.right_hand_label.config(text="Right Hand:")
            self.left_hand_label.config(text="Left Hand:")
            self.right_hand_xcoord_label.config(text = "Right Hand X Coordinates:")
            self.right_hand_ycoord_label.config(text = "Right Hand Y Coordinates:")
            
            self.right_hand_output_label.configure(text = "Right Gesture Output:")
            self.left_hand_output_label.configure(text = "Left Gesture Output:")
            self.vertical_output_label.configure(text = "Vertical Position Output:")
            self.horizontal_hand_output_label.configure(text = "Horizontal Position Output:")
        
        # Call the update function after 10 milliseconds
        self.window.after(10, self.update)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def main():
    root = tk.Tk()
    app = App(root, "Hand Gesture Control System")
    root.mainloop()

if __name__ == "__main__":
    main()
