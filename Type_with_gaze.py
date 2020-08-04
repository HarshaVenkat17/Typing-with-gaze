#import required libraries
import cv2
import numpy as np
import dlib
from pygame import mixer
import time

#allocating space to each letter on keyboard and turn them on
def draw_letters(letter_index, text, letter_on):
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 150
    elif letter_index == 6:
        x = 200
        y = 150
    elif letter_index == 7:
        x = 400
        y = 150
    elif letter_index == 8:
        x = 600
        y = 150
    elif letter_index == 9:
        x = 800
        y = 150
    elif letter_index == 10:
        x = 0
        y = 300
    elif letter_index == 11:
        x = 200
        y = 300
    elif letter_index == 12:
        x = 400
        y = 300
    elif letter_index == 13:
        x = 600
        y = 300
    elif letter_index == 14:
        x = 800
        y = 300
    elif letter_index == 15:
        x = 0
        y = 450
    elif letter_index == 16:
        x = 200
        y = 450
    elif letter_index == 17:
        x = 400
        y = 450
    elif letter_index == 18:
        x = 600
        y = 450
    elif letter_index == 19:
        x = 800
        y = 450
    #drawing rectangles for each letter and centering them
    width = 200
    height = 150
    space =3
    text_width,text_height= cv2.getTextSize(text,font,10,4)[0]
    text_x = int((width - text_width) / 2) + x
    text_y = int((height + text_height) / 2) + y
    #each letter will be highlighted if it is active else, it will be dark
    if letter_on:
        cv2.rectangle(keyboard, (x + space, y + space), (x + width - space, y + height - space), (255,255, 255),-1)
        cv2.putText(keyboard, text, (text_x, text_y), font,10, (242,191,7),6)
    else:
        cv2.rectangle(keyboard, (x + space, y + space), (x + width - space, y + height - space), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y),font,10, (255, 255, 255),4)

#create main menu
def menu():
    rows, cols = keyboard.shape[0:2]
    th_lines = 4 
    cv2.line(keyboard, (int(cols/2) - int(th_lines/2), 0),(int(cols/2) - int(th_lines/2), rows),(51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (70, 325), font,8, (255, 255, 255), 4)
    cv2.putText(keyboard, "RIGHT", (70 + int(cols/2), 325), font,8, (255, 255, 255), 4)

#helper function to find mid-point of 2 points
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

#helper function to find lengths of lines joining eye-extremes 
def hypot(a,b):
    return (a**2+b**2)**0.5

#helper function to tell if person is blinking by calculating ratio of lengths found
def blink_ratio(eye_points, landmarks):
    left = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
    right = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)
    center_top = midpoint(landmarks.part(eye_points[1]), landmarks.part(eye_points[2]))
    center_bottom = midpoint(landmarks.part(eye_points[5]), landmarks.part(eye_points[4]))
    hor_length = hypot((left[0] - right[0]),(left[1] - right[1]))
    ver_length = hypot((center_top[0] - center_bottom[0]),(center_top[1] - center_bottom[1]))
    ratio = hor_length/ver_length
    return ratio

#helper function to color boundaries of eyes(returns array of points corresponding to eyes)  
def eyes_contour(landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

#returns gaze-ratio to find out which side the person is looking 
def get_gaze_ratio(eye_points, landmarks):
    eye_region = np.array([(landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y),
                                (landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y),
                                (landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y),
                                (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y),
                                (landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y),
                                (landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)], np.int32)

    #finding region of eye
    height, width= frame.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    
    #calculating area of white region visible in the eye and corresponding gaze ratio.
    gray_eye = eye[min_y: max_y, min_x: max_x]
    thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)[1]
    height, width = thresh_eye.shape
    left_thresh = thresh_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_thresh)

    right_thresh = thresh_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_thresh)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

#play sound for indicating selection of keyboard and letter
def load_sound(c):
    mixer.init()
    if c==1:
        mixer.music.load('sound.wav')
    elif c==2:
        mixer.music.load('left.wav')
    elif c==3:
        mixer.music.load('right.wav')
    mixer.music.play()

#open a file to write
f=input(r"Enter file address(press Enter to skip): ")
mark=1
try:
    fp=open(f,"a+")
except:
    mark=0
 
capture = cv2.VideoCapture(0)
#create a white board to put all the text
whiteboard = np.zeros((300, 1000), np.uint8)
whiteboard[:] = 255

#initialise face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#creating a virtual keyboard
keyboard = np.zeros((600, 1000, 3), np.uint8)
keys_set_1 = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5",
              5: "Q", 6: "W", 7: "E", 8: "R", 9: "T",
              10: "A", 11: "S", 12: "D", 13: "F", 14: "G",
              15: "Z", 16: "X", 17: "C", 18: "V", 19: ">"
              }
keys_set_2 = {0: "6", 1: "7", 2: "8", 3: "9", 4: "0",
              5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
              10: "H",11: "J",12: "K",13: "L",14: "sp",
              15: "V",16: "B",17: "N",18: "M",19: "<"
              }


# Counters and limits
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 5
frames_active_letter = 18

#Initialisation of variables for storing text,selecting keyboard and font
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_menu = True
keyboard_selection_frames = 0
font=cv2.FONT_HERSHEY_PLAIN

#start processing each frame
while True:
    frame = capture.read()[1]
    rows, cols= frame.shape[0:2]
    keyboard[:] = (26, 26, 26)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create some white space for loading bar
    frame[rows - 50: rows, 0: cols] = (255,255,255)
    # Open menu
    if select_menu is True:
        menu()

    # Display keys according to keyboard selected
    if keyboard_selected == "left":
        keys_set = keys_set_1
    else:
        keys_set = keys_set_2
    active_letter = keys_set[letter_index]

    # Face detection
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye, right_eye = eyes_contour(landmarks)

        # Calculate blink ratio to detect blinking
        left_eye_ratio = blink_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = blink_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Creating a red boundary around eyes
        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)

        if select_menu is True:
            # Detecting gaze ratio to select keyboard
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            #selecting right keyboard if gaze_ratio <= 0.9 and left otherwise
            if gaze_ratio <= 0.9:
                keyboard_selected = "right"
                keyboard_selection_frames += 1
                if keyboard_selection_frames == 15:
                    select_menu = False
                    load_sound(3)
                    frames = 0
                    keyboard_selection_frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0
            else:
                keyboard_selected = "left"
                keyboard_selection_frames += 1
                if keyboard_selection_frames == 15:
                    select_menu = False
                    load_sound(2)
                    frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0
        else:
            # Selecting keys based on blink ratio when they are active
            if blinking_ratio > 5:
                blinking_frames += 1
                frames -= 1

                # Color the eyes with green when blinking
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

                # Select a letter and append to text
                if blinking_frames == frames_to_blink:
                    if active_letter != "<" and active_letter != "sp":
                        text += active_letter
                    if active_letter == "sp":
                        text += " "
                    load_sound(1)
                    select_menu = True
            else:
                blinking_frames = 0

    # Display and highlight letters on the keyboard
    if select_menu is False:
        if frames == frames_active_letter:
            letter_index += 1
            frames = 0
        if letter_index == 20:
            letter_index = 0
        for i in range(20):
            if i == letter_index:
                light = True
            else:
                light = False
            draw_letters(i, keys_set[i], light)

    # Display selected letter on whiteboard
    cv2.putText(whiteboard, text, (80, 100), font, 9, 0, 3)
    
    # Show loading
    load= blinking_frames / frames_to_blink
    loading_x = int(cols * load)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
    cv2.putText(frame, "LOADING...", (190,475), font, 4,(255,255,255), thickness=3)

    # Display video capture, keyboard and whiteboard
    cv2.imshow("Frame", frame)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Write here", whiteboard)
    # Press "esc" to close
    key = cv2.waitKey(1)
    if key == 27:
        break
    
# close any open files
if mark:
    fp.write(text+"\n")
    fp.close()
    
capture.release()
cv2.destroyAllWindows()
exit()
