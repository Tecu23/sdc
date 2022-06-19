import cv2
import numpy as np
import math

"""
    Metoda pentru calcularea distantei dintre 2 puncte
        IN: Cele 2 puncte pentru care vom calcula distanta
        OUT: Distanta respectiva dintre cele 2 puncte
"""
def distance(a,b):
    return math.sqrt( ( (a[1]-b[1])**2 ) + ( (a[0]-b[0])**2 ) )

"""
    Metoda pentru aproximarea dintre centrele ale 2 contururi
        IN: 2 contururi pentru care vom calcula distanta centrelor
        OUT: Distanta centrelor si pozitiile centrelor
"""
def distance_between_centers(cnt_a,cnt_b):
    
    # 1. Calcularea centrului conturului punctului a
    center_of_contour_a = cv2.moments(cnt_a)
    center_x_a = int(center_of_contour_a["m10"] / center_of_contour_a["m00"])
    center_y_a = int(center_of_contour_a["m01"] / center_of_contour_a["m00"])

    # 2. Calcularea centrului conturului punctului b
    center_of_contour_b = cv2.moments(cnt_b)
    center_x_b = int(center_of_contour_b["m10"] / center_of_contour_b["m00"])
    center_y_b = int(center_of_contour_b["m01"] / center_of_contour_b["m00"])
    
    # 3. Calcularea distantei dintre cele 2 centre
    minDist = distance((center_x_a,center_y_a),(center_x_b,center_y_b))
    contour_a=(center_x_a,center_y_a)
    contour_b=(center_x_b,center_y_b)
    
    return minDist,contour_a,contour_b
"""
    Metoda pentru gasirea celui mai mare contur din imagine, mai mare decat aria minima
        IN: Imaginea in nuante de gri si aria minima a contururilor
        OUT: Imaginea care poate avea fie un contur ramas, fie imaginea neagra, daca nu avem niciun contur si o variabila bool in cazul in care am gasit un contur
"""
def largest_contour(gray):
    found = False
    image = np.zeros(gray.shape, dtype=gray.dtype)

    # 1. Transformarea imaginii intr-o imagine binara
    bin_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # 2. Gasirea tuturor contururilor din imagine
    cnts = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    area_max = 0
    id_max = -1

    # 3. Enumeram printre contururile gasite si il cautam pe cal cu cea mai mare arie
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > area_max:
            area_max = area
            id_max = index
            found = True
    
    # 4. Daca exista un contur de arie maxima il aplicam imaginii
    if id_max != -1:
        image = cv2.drawContours(image, cnts, id_max, (255,255,255), -1)

    return image, found

"""
    Metoda pentru estimarea benzii de mijloc
        IN: Imaginea cu banda de mijloc si distanta maxima dintre contururi
        OUT: Imaginea reprezentand banda de mijloc estimata
"""
def estimate_midlane(image,MaxDistance):
    image_zero = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # 1. Gasirea tuturor contururilor din imagine
    cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # 2. Pastram toate contururile care nu sunt linii
    min_area = 1
    contours = []
    for index, _ in enumerate(cnts):
        area = cv2.contourArea(cnts[index])
        if area > min_area:
            contours.append(cnts[index])
    cnts = contours

    # [Best Match for contour 0, Best Match for contour 0, ...]
    contour_id_match = [] 
    
    # 3. Conectarea tuturor contururilor de distanta minima
    for index, cnt in enumerate(cnts):
        prev_min_distance = 100000
        best_index = 0
        best_contour_a = 0
        best_contour_b = 0
        
        for index_contour in range(len(cnts) - index):
            index_contour = index_contour + index
            contour = cnts[index_contour]
            
            if index != index_contour:
                min_dist, contour_a, contour_b  = distance_between_centers(cnt, contour)

                if min_dist < prev_min_distance:
                    if len(contour_id_match) == 0:
                        prev_min_distance = min_dist
                        best_index = index_contour
                        best_contour_a = contour_a
                        best_contour_b = contour_b   

                    else:
                        Present = False
                        
                        for i in range(len(contour_id_match)):
                            if index_contour == i and index == contour_id_match[i]:
                                Present = True
                        
                        if not Present:
                            prev_min_distance = min_dist
                            best_index = index_contour
                            best_contour_a = contour_a
                            best_contour_b = contour_b   
        
        if prev_min_distance != 100000 and prev_min_distance > MaxDistance:
            break

        if type(best_contour_a) != int:
            contour_id_match.append(best_index)
            cv2.line(image_zero,best_contour_a,best_contour_b,(0,255,0),thickness=2)
    
    image_zero = cv2.cvtColor(image_zero,cv2.COLOR_BGR2GRAY)

    # 4. Gasim estimarea benzii de mijloc prin gasirea celui mai mare contur din imagine 
    image_largest,found = largest_contour(image_zero)

    # 5. Daca am gasit cel mai mare contur, il intoarcem, altfel intoarcem imaginea initiala
    if(found):
        return image_largest
    else:
        return image