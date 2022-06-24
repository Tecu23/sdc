import tensorflow as tf
from keras.models import load_model
import os
import cv2
import numpy as np

from sign_tracking import SignTracking

detected_img = 0 #Set this to current dataset images size so that new images number starts from there and dont overwrite

write_data = False
draw_detected = True
display_images = False
model_loaded = False
model = 0

sign_classes = ["speed_sign_30", "speed_sign_50", "stop", "no_sign"]

sign_track = SignTracking()

"""
    Metoda pentru transformarea unei imagini intr-o imagine ce poate fi clasificata
        IN: Imaginea initiala sub o forma oarecare
        OUT: Imaginea modificata
"""
def image_forKeras(image):

    # 1. Transformare din BGR in RGB
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # 2. Modificaarea dimensiunilor imaginii
    image = cv2.resize(image,(30,30))

    # 3. Modificare dimensiuni model
    image = np.expand_dims(image, axis=0)

    return image


"""
    Metoda pentru localizarea, clasificarea si urmarirea semnelor de circulatie dintr-o imagine
        IN: Imaginea gri, Copie a imaginii initiale, Imaginea pe care vom desena si modelul retelei
        OUT: Imaginea desenata
"""
def sign_detection_tracking(gray,cimg,frame_draw,model):
    
    # 3. Daca modelul este de detectie, atunci cautam in imagine semnele de circulatie
    if sign_track.mode == "Detection":
        cv2.putText(frame_draw,str(sign_track.tracked_class),(10,80),cv2.FONT_HERSHEY_PLAIN,0.7,(255,255,255),1)
        number_votes_circle = 40 
        canny_threshold = 200 
        min_distance_circles = 100 
        max_rad = 150 

        # 4. Localizarea semnelor folosing HoughCircles
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT, 1, min_distance_circles, param1=canny_threshold, param2=number_votes_circle, minRadius=10, maxRadius=max_rad)

        # 4a. Verificam daca am gasit cercuri
        if circles is not None:
            circles = np.uint16(np.around(circles))

            # 4b. Trecem prin fiecare cerc localizat
            for i in circles[0,:]:

                center =(i[0],i[1])
                match_found,match_idx = sign_track.match_center(center)

                radius = i[2] + 5
                if (radius !=5):
                    global detected_img
                    detected_img = detected_img + 1 

                    startP = (center[0]-radius,center[1]-radius)
                    endP = (center[0]+radius,center[1]+radius)
                    
                    # 4c. Extragem regiunea de interes a cercului
                    detected_sign = cimg[startP[1]:endP[1],startP[0]:endP[0]]

                    # 4d. Clasificam acest regiune de interes intr-un semn de circulatie
                    if(detected_sign.shape[1] and detected_sign.shape[0]):
                        sign = sign_classes[np.argmax(model(image_forKeras(detected_sign)))]

                        # 4e. Verificam daca cercul este sau nu un semn
                        if(sign != "No_Sign"):

                            # 4f. IDaca am gasit un semn crestem increderea
                            if match_found:
                                sign_track.known_centers_confidence[match_idx] += 1

                                # 4g. Daca acelasi semn este detectat de 3 ori atunci trecem la modul Tracking
                                if(sign_track.known_centers_confidence[match_idx] > 3):
                                    circle_mask = np.zeros_like(gray)
                                    circle_mask[startP[1]:endP[1],startP[0]:endP[0]] = 255
                                    sign_track.mode = "Tracking" 
                                    sign_track.tracked_class = sign
                                    sign_track.old_gray = gray.copy()
                                    sign_track.corners = cv2.goodFeaturesToTrack(sign_track.old_gray, mask=circle_mask, **sign_track.feature_params)
                                    sign_track.mask = np.zeros_like(frame_draw)

                            # 4h. Altfel modificam locatia semnului si de cate ori a fost gasit
                            else:
                                sign_track.known_centers.append(center)
                                sign_track.known_centers_confidence.append(1)

                            # 4i. Afisam ce am detectat                      
                            cv2.putText(frame_draw,sign,(endP[0]-20,startP[1]+10),cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,255),1)
                            if draw_detected:
                                cv2.circle(frame_draw,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
                                cv2.circle(frame_draw,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

    # 5. Daca modulul este de tracking
    else:
        # Calculam fluxul optic
        p1, st, err = cv2.calcOpticalFlowPyrLK(sign_track.old_gray, gray, sign_track.corners, None,**sign_track.lk_params)
        
        # 5a. Daca nu am gasit fluxul, cautam puncte noi
        if p1 is None:
            sign_track.mode = "Detection"
            sign_track.mask = np.zeros_like(frame_draw)
            sign_track.Reset()

        # 5b. Dar daca am gasit modificam variabilele
        else:
            # Selectam puncte
            good_new = p1[st == 1]
            good_old = sign_track.corners[st == 1]
            # Sesenam noile urme
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = (int(x) for x in new.ravel())
                c, d = (int(x) for x in old.ravel())
                sign_track.mask = cv2.line(sign_track.mask, (a, b), (c, d), sign_track.color[i].tolist(), 2)
                frame_draw = cv2.circle(frame_draw, (a, b), 5, sign_track.color[i].tolist(), -1)
            frame_draw_ = frame_draw + sign_track.mask 
            np.copyto(frame_draw,frame_draw_) 
            sign_track.old_gray = gray.copy()
            sign_track.corners = good_new.reshape(-1, 1, 2)           

    return frame_draw


def detect_Signs(frame,frame_draw):
    
    global model_loaded

    if not model_loaded:

        # 1. Incarcam modelul
        global model
        model = load_model('model/model.h5', compile=False)
        model.summary()
        model_loaded = True

    # 2. Transformam imaginea din RGB in nuante de gri
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.putText(frame_draw, sign_track.mode, (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 1)

    frame_draw = sign_detection_tracking(gray.copy(), frame.copy(), frame_draw, model)

    return sign_track.mode , sign_track.tracked_class, frame_draw