#worked on by Naveed Sekender
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.layers import LayerNormalization
from imutils.video import VideoStream
import cv2
import sys
import numpy as np
import keras
import app
from keras.models import load_model

#this is a thingy that is for the ltters n stuff 
alphabet = { #any new gesture has to be added here
    0:"A",1:"B",2:"C",3:"D", 4:"E",	5:"F",	6:"G",	7:"H",	8:"I", 9:"J",10:"K", 11:"L", 12:"M", 13:"N",
	14:"O",	15:"P",	16:"Q",	17:"R", 18:"S",	19:"T",	20:"U",	21:"V",22:"W",23:"X",24:"Y",25:"Z",
}

#this is so return the letter 
def return_value(letter):
    return(letter)

#this is a null function
def nothing(x):
    pass

#this is to crop stuff
def take_pics(image, x, y, width, height):
    return image[y:y + height, x:x + width]

#this is to get the stuff from the dictionary
def get_class_label(val, alphabet):
    for key, value in alphabet.items():
        if value == val:
            return key


#load up my model
model = load_model('dataset.h5')
SENTENCE = ""

video_capture = cv2.VideoCapture(0) 
cv2.namedWindow('Model Image')

# set the ration of main video screen
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

# set track bar of threshold values for Canny edge detection
# more on Canny edge detection here:
cv2.createTrackbar('lower_threshold', 'Model Image', 0, 255, nothing)
cv2.createTrackbar('upper_threshold', 'Model Image', 0, 255, nothing)
cv2.setTrackbarPos('lower_threshold', 'Model Image', 100)
cv2.setTrackbarPos('upper_threshold', 'Model Image', 0)

# VARIABLES INITIALIZATION
# THRESHOLD - ratio of the same letter in the last N_FRAMES predicted letters
THRESHOLD = 0.9
N_FRAMES = 30

IMG_SIZE = 50
letter = [] # temporary letter
LETTERS = np.array([], dtype='object') # array with predicted letters

START = 0 # start/pause controller


while True:
    blank_image = np.zeros((100,800,3), np.uint8) # black image for the output
    ret, frame = video_capture.read() # capture frame-by-frame
    # set the corners for the square to initialize the model picture frame
    x_0 = int(frame.shape[1] * 0.1)
    y_0 = int(frame.shape[0] * 0.25)
    x_1 = int(x_0 + 200)
    y_1 = int(y_0 + 200)

    # MODEL IMAGE INITIALIZATION
    hand = take_pics(frame,300,300,300,300)
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = cv2.erode(blurred, None, iterations=2)
    blurred = cv2.dilate(blurred, None, iterations=2)
    lower = cv2.getTrackbarPos('lower_threshold', 'Model Image')
    upper = cv2.getTrackbarPos('upper_threshold', 'Model Image')
    edged = cv2.Canny(blurred,lower,upper) # aplly edge detector
    model_image = ~edged 
    model_image = cv2.resize(
        model_image,
        dsize=(IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC
    )


    model_image = np.array(model_image)
    model_image = model_image.astype('float32') / 255.0

    #try to see if it works or not
    try:
        model_image = model_image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        predict = model.predict(model_image)
        for values in predict:
            if np.all(values < 0.3):
                # if probability of each class is less than .5 return a message
                letter = 'Cannot classify'
            else:
                predict = np.argmax(predict, axis=1) + 1
                letter = get_class_label(predict, alphabet)
                return_value(letter)
                LETTERS = np.append(LETTERS, letter)
    except:
        pass

#
    # TEXT INITIALIZATION
    cv2.putText(img=frame, text = "", org=(x_0+140,y_0+195), fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,0,255), fontScale=1)
    cv2.putText(img=frame,text=letter, org=(x_0+10,y_0+20),fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255), fontScale=1 )
    cv2.putText(img=blank_image, text='Result: ' + SENTENCE, org=(10, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=1, color=(0,0,255),fontScale=1)

    # draw rectangle for hand placement
    cv2.rectangle(frame, (x_0, y_0), (x_1, y_1), (0, 255, 255), 2)

    # display the resulting frames
    cv2.imshow('Main Image', frame)
    cv2.imshow('Model Image', edged)
    cv2.imshow('Output', blank_image)

    #define key 
    key = cv2.waitKey(1) & 0xFF
	# if the `g` key was pressed, break from the loop
    if key == ord("g"):
       break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
