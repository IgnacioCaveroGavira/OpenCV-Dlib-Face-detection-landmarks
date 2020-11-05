# importing the necessary packages
import cv2
import dlib
import numpy as np


#---------------------- Config ----------------------
Enable_landmarks = False

MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

RED = (0, 0, 255)
GREEN = (0,255,0)

windows_name = "Virtual Makeup"

# Face areas
LANDMARK_AREA_UPPER_LIP = [48,49,50,51,52,53,54,64,63,62,61,60,48]
LANDMARK_AREA_LOWER_LIP = [48,60,67,66,65,64,54,55,56,57,58,59,48]

LANDMARK_LEFT_IMAGE_EYE = [[36,39], [38,40]]
LANDMARK_RIGHT_IMAGE_EYE = [[42,45],[43,47]]

original = cv2.imread("sources/Face2.jpg")
frame_for_face_detection = original.copy()

original = np.float32(original)/255

glasse1 = cv2.imread("sources/Glasse1.png", cv2.IMREAD_UNCHANGED)
glasse2 = cv2.imread("sources/Glasse2.png", cv2.IMREAD_UNCHANGED)
glasse3 = cv2.imread("sources/Glasse3.png", cv2.IMREAD_UNCHANGED)
glasse4 = cv2.imread("sources/Glasse4.png", cv2.IMREAD_UNCHANGED)

glasse1 = np.float32(glasse1)/255
glasse2 = np.float32(glasse2)/255
glasse3 = np.float32(glasse3)/255
glasse4 = np.float32(glasse4)/255

glasses = [glasse1,glasse2,glasse3,glasse4]


face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(MODEL_PATH)
#---------------------------------------------------


#---------------------- defs ----------------------

# Draw Dlib landmarks in a image
def draw_landmars(image, landmarks):
    for n in range(0, landmarks.num_parts):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def get_landmark_area(landmarks, point_numbers):
    res = []

    for e in point_numbers:
        point = landmarks.part(e)
        res.append([point.x, point.y])

    return [np.array(res)]


def mask_creator(img, areas, inversed = False):
    
    mask = np.zeros(img.shape, img.dtype)   
    alpha_background = 1
    alpha_areas = (0,0,0)
    
    if inversed:
        alpha_background = 0
        alpha_areas = (1,1,1)


    b_channel, g_channel, r_channel = cv2.split(mask)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha_background

    for area in areas:
        cv2.fillPoly(alpha_channel, area, alpha_areas)
        
    mask = cv2.merge((alpha_channel, alpha_channel, alpha_channel))

    return mask


def lips_coloring(img, landmarks, color):
    image = img.copy()

    upper_lip = get_landmark_area(landmarks, LANDMARK_AREA_UPPER_LIP)
    lower_lip = get_landmark_area(landmarks, LANDMARK_AREA_LOWER_LIP)

    mask_get_only_lips = mask_creator(image, [upper_lip, lower_lip])
    mask_get_only_lips = cv2.GaussianBlur(mask_get_only_lips,(5,5),0)
    
    image_colored = image.copy()

    if color == 1:
        image_colored[:, :, 1] = 0
        image_colored[:, :, 2] = 0

    if color == 2:
        image_colored[:, :, 0] = 0
        image_colored[:, :, 2] = 0

    if color == 3:
        image_colored[:, :, 0] = 0
        image_colored[:, :, 1] = 0


    lips_colored = cv2.multiply(image_colored,(1 - mask_get_only_lips ))


    mask_delete_lips = mask_creator(image, [upper_lip, lower_lip], True)
    mask_delete_lips = cv2.GaussianBlur(mask_delete_lips,(5,5),0)

    image  = cv2.multiply(image,(1 - mask_delete_lips ))

    image = cv2.add(image, lips_colored)

    return image


def put_glasse(face_image, image_glasse, landmarks):
    
    glasse_resized = getGlassesProporcional(face_image, image_glasse, landmarks)

    b_channel, g_channel, r_channel, alpha_channel = cv2.split(glasse_resized)    

    alpha_mask = cv2.merge((alpha_channel, alpha_channel, alpha_channel,alpha_channel)) 

    #"Clean" BGR channels from glasse image (clean means put 0 in BGR channels outside of alpha channel shade)
    glasse_resized = cv2.multiply(glasse_resized,alpha_mask)   

    # Removed alpha_channel  
    alpha_mask = alpha_mask[:,:,:3]
    glasse_resized = glasse_resized[:,:,:3]

    height, width, channels = glasse_resized.shape 

    p = landmarks.part(28)
    Ox = p.y - np.uint16(height/2)
    Oy = p.x - np.uint16(width/2)

    # Mask to delete glasse in face image
    mask = np.zeros(face_image.shape, face_image.dtype) 
    alpha_mask = cv2.merge((alpha_channel, alpha_channel, alpha_channel))
    mask[Ox:(Ox + height),Oy:(Oy + width)]= alpha_mask

    # Glasse imagen with face image size
    glasse_in_position = np.zeros(face_image.shape, face_image.dtype) 
    glasse_in_position[Ox:(Ox + height),Oy:(Oy + width)]= glasse_resized

    face_masked = cv2.multiply(face_image,(1 - mask))

    image = cv2.add(face_masked, glasse_in_position)

    return image


def getGlassesProporcional(face_image, image_glasse, landmarks):
    p0 = landmarks.part(0)
    p16 = landmarks.part(16)
    height, width, channels = image_glasse.shape 
    face_width = p16.x - p0.x

    prec = (face_width / width)
    
    glasse_resized = cv2.resize(image_glasse, None, fx=prec, fy=prec, interpolation = cv2.INTER_CUBIC  )

    return  glasse_resized


def onChange_glasses(value):
    # global original
    # global landmarks
    res = original

    bar_lips_value = cv2.getTrackbarPos('Lips color', windows_name)

    if value != 0:
        res = put_glasse(res, glasses[value - 1], landmarks)
    
    if bar_lips_value != 0:
        res = lips_coloring(res, landmarks, bar_lips_value)

    
    cv2.imshow(windows_name, res)


def onChange_lips(value):
    # global original
    # global landmarks
    res = original

    bar_glasse_value = cv2.getTrackbarPos('Glasses', windows_name)

    if bar_glasse_value != 0:
        res = put_glasse(res, glasses[bar_glasse_value - 1], landmarks)
            
    if value != 0:
        res = lips_coloring(res, landmarks, value)

    cv2.imshow(windows_name, res)

#--------------------------------------------------


#---------------------- Main ----------------------

img_gray = cv2.cvtColor(frame_for_face_detection, cv2.COLOR_BGR2GRAY)

faces = face_detector(img_gray)

landmarks = landmark_detector(img_gray, faces[0]) #Just one face will be used

if Enable_landmarks:            
    draw_landmars(original,landmarks)

cv2.namedWindow(windows_name, cv2.WINDOW_AUTOSIZE)
cv2.imshow(windows_name, original)

cv2.createTrackbar("Lips color", windows_name, 0, 3,onChange_lips)

cv2.createTrackbar("Glasses", windows_name, 0, 4,onChange_glasses)

cv2.waitKey(0)
