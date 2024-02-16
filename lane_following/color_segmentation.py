import cv2
import numpy as np

# Constante pentru colori
hue_low = 50#89
lit_low = 51#88
sat_low = 11#36

hue_low_y = 24#26
hue_high_y = 32#38
lit_low_y = 38#40
sat_low_y = 106#48

"""
    Metoda pentru aplicarea segmentarii de culoare
        IN: Imaginea HLS, gama de valori inferioara si gama de valori superioara
        OUT: Imaginea segmentata dupa masca si dilatata
"""
def mask_segmentation(frame_hls, lower, upper):

    # Aplicarea mastii de culoare
    mask = cv2.inRange(frame_hls, lower, upper)

    # Aplicarea unui proces de dilatare a imaginii pentru a mari obiectele din imagine
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (4,4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    return mask

"""
    Metoda pentru extragerea obiectelor din imagine cu o dimensiune mai mica de min_area
        IN: Imaginea si valoarea ariei minime
        OUT: Imagine avand obiectele mai mici decat aria minima extrase
"""
def remove_smaller_contours(image, min_area):
    small_contours = []

    # 1. Transformam imaginea intr-o imagine binara
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

    # 2. Gasim contururile tuturor obiectelor din imagine
    cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 3. Enumeram prin toate contururile si in cazul in care au o arie mai mica decat aria minima, le adaugam in lista pentru ariile mici
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area < min_area:
            small_contours.append(cnt)
    
    # 4. Toate contururile gasite le coloram cu negru pentru a le extrage din imaginea binara
    image = cv2.drawContours(image, small_contours, -1, 0, -1)
            
    return image

"""
    Metoda pentru gasirea celui mai mare contur din imagine, mai mare decat aria minima
        IN: Imaginea in nuante de gri si aria minima a contururilor
        OUT: Imaginea care poate avea fie un contur ramas, fie imaginea neagra, daca nu avem niciun contur si o variabila bool in cazul in care am gasit un contur
"""
def largest_contour(gray, min_area):
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
    
    # 4. Daca conturul gasit are aria mai mica decat min_area atunci intoarcem imaginea goala, altfel coloram conturul cu alb
    if area_max < min_area:
        found = False
    else:
        image = cv2.drawContours(image, cnts, id_max, (255,255,255), -1)

    return image, found

"""
    Metoda pentru gasirea extremelor unui obiect
        IN: Imaginea cu obiectul respectiv
        OUT: Extremele de jos si sus ale obiectului sau imaginea neagra, in cazul in care nu exista un obiect in imagine
"""
def find_ext(image):
    pos = np.nonzero(image)
    return (0,0) if len(pos) == 0 else (pos[0].min(),pos[0].max())

"""
    Metoda pentru extragerea unei zone de interes dintr-o imagine bazate pe un punct de inceput si un punct de sfarsit
        IN: Imaginea, punctul de inceput si punctul de sfarsit
        OUT: Imaginea cu regiunea respectiv extrasa
"""
def extract_region(image,start_point,end_point):
    region = np.zeros(image.shape, dtype=np.uint8)

    # 1. Cream un dreptunghi folosind cele 2 puncte
    cv2.rectangle(region, start_point, end_point, 255, thickness=-1)
    
    # 2. Returnam Imaginea cu regiunea de interes
    return cv2.bitwise_and(image,region)

"""
    Metoda pentru extragerea punctului de minim al unui obiect dintr-o imagine, dintr-un anumit rand
        IN: Imaginea respectiva si randul respectiv
        OUT: Punctul de minim 
"""
def extract_point(image,specified_row):
    
    p = (0 , specified_row)
    specified_row_data = image[specified_row-1, :]
    pos = np.nonzero(specified_row_data)

    if len(pos[0]) != 0:
        min_col = pos[0].min()
        p = (min_col,specified_row)
    return p

"""
    Metoda pentru gasirea marginii de jos a benzii
        IN: Imaginea cu contrururile
        OUT: Pozitia marginii
"""
def find_low_row(image):
    pos = np.nonzero(image)

    return image.shape[0] if len(pos) == 0 else pos[0].max()

"""
    Metoda pentru gasirea marginilor si punctelor inferioare extreme ale benzii
        IN: Imagine cu banda exterioara de arie maxima
        OUT: Imagine cu marginile benzii exterioara si lista cu punctele extreme inferioare acesteia
"""
def points_lowest(gray):
    
    points_outerlane=[]
    image = np.zeros(gray.shape, dtype=gray.dtype)
    lane_one_side=np.zeros(gray.shape, dtype=gray.dtype)
    lane_two_side=np.zeros(gray.shape, dtype=gray.dtype)

    # 1. Transformarea imaginii intr-o imagine binara
    bin_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # 2. Gasim contururile tuturor obiectelor din imagine
    cnts = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    image = cv2.drawContours(image, cnts, 0, (255,255,255), 1)

    # 3. Gasirea marginilor externe ale benzii
    top,bot = find_ext(image)

    # 4. Extragerea portiunii de contur al benzii
    Contour_TopBot_PortionCut = extract_region(image, (0, top + 25), (image.shape[1], bot-15))

    # 5. Gasirea conturilor marginilor benzii pentru a gasi punctele extreme
    cnts2 = cv2.findContours(Contour_TopBot_PortionCut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    low_x=-1
    low_y=-1
    
    comp_row=0

    line = np.copy(lane_one_side)
    cnts_tmp = []

    if(len(cnts2)>2):
        for index_tmp, cnt_tmp in enumerate(cnts2):
            if((cnt_tmp.shape[0])>50):
                cnts_tmp.append(cnt_tmp)
        cnts2 = cnts_tmp

    for index, cnt in enumerate(cnts2):
        lane_one_side = np.zeros(gray.shape,dtype=gray.dtype)
        lane_one_side = cv2.drawContours(lane_one_side, cnts2, index, (255,255,255), 2)
        lane_two_side = cv2.drawContours(lane_two_side, cnts2, index, (255,255,255), 2)

        if len(cnts2) == 2:
            if index == 0:
                line = np.copy(lane_one_side)
                low_x = find_low_row(lane_one_side)
            elif index == 1:
                low_y = find_low_row(lane_one_side)
                if low_x<low_y:
                    comp_row = low_x
                else:
                    comp_row = low_y
                p_a = extract_point(line,comp_row)
                p_b = extract_point(lane_one_side,comp_row)
                points_outerlane.append(p_a)
                points_outerlane.append(p_b)
    
    return lane_two_side, points_outerlane

"""
    Metoda pentru extragerea benzii exterioare din imagine
        IN: Imaginea initiala, imaginea segmentata si aria minima de detecterare a unui obiect
        OUT: Imagine reprezentand banda exterioara si Punctele inferioare extreme ale benzii
"""
def outer_lane(frame, mask, min_area):
    points_outerlane=[]

    # 1. Aplicarea unei operatii de AND intre imaginea initiala si imaginea segmentata pentru a extrage partea RGB a regiunii segmentate
    lane_rgb = cv2.bitwise_and(frame, frame, mask = mask)

    # 2. Convertirea imaginii din RGB in nunate de gri pentru a putea gasi conturile obiectelor din imagine
    lane_gray = cv2.cvtColor(lane_rgb, cv2.COLOR_BGR2GRAY)

    # 3. Pastrarea obiectelor mai mari decat min_area din imagine
    lane_gray_bigger_contours = remove_smaller_contours(lane_gray, min_area)

    # 4. Aplicarea unei operatii AND intre imaginea cu nuante de gri a benzii si imaginea cu conturile mai mari pentru a pastra obiectele mai mari in nuanta de gri
    lane_gray = cv2.bitwise_and(lane_gray, lane_gray_bigger_contours)

    # 5. Pastrarea numai celui mai mare contour din imagine
    outerlane_largest,found = largest_contour(lane_gray_bigger_contours, min_area)

    if(found):
        # 5a. Daca a fost gasit cel mai mare contur atunci calculam valoarea marginii benzii de jos si punctele extreme ale benzii
        lane, points_outerlane = points_lowest(outerlane_largest)
    else:
        # 5b. Daca nu a fost gasit cel mai mare contur atunci intoarcem imaginea neagra
        lane = np.zeros(lane_gray.shape, lane_gray.dtype)

    return lane,points_outerlane

"""
    Metoda pentru extragerea benzii de mijloc din imagine
        IN: Imaginea initiala, imaginea segmentata si aria minima de detecterare a unui obiect
        OUT: Imagine reprezentand marginile benzii de mijloc
"""
def mid_lane(frame, mask, min_area):
    
    # 1. Aplicarea unei operatii de AND intre imaginea initiala si imaginea segmentata pentru a extrage partea RGB a regiunii segmentate
    lane_rgb = cv2.bitwise_and(frame, frame, mask=mask)

    # 2. Convertirea imaginii din RGB in nunate de gri pentru a putea gasi conturile obiectelor din imagine
    lane_gray = cv2.cvtColor(lane_rgb,cv2.COLOR_BGR2GRAY)

    # 3. Pastrarea obiectelor mai mari decat min_area din imagine
    lane_gray_bigger_contours = remove_smaller_contours(lane_gray,min_area)
    
    # 4. Aplicarea unei operatii AND intre imaginea cu nuante de gri a benzii si imaginea cu conturile mai mari pentru a pastra obiectele mai mari in nuanta de gri
    lane_gray = cv2.bitwise_and(lane_gray,lane_gray_bigger_contours)
    
    # 5. Aplicarea detectiei marginilor
    lane_gray_smoothed = cv2.GaussianBlur(lane_gray, (11,11), 1)
    lane_edge = cv2.Canny(lane_gray_smoothed, 50, 150, None, 3)

    return lane_edge

"""
    Metoda pentru segmentarea imaginii initiale gasind banda exterioara si cea de mijloc
        IN: Imaginea initiala si aria minima de detectare a unui obiect
        OUT: Imagine reprezentand marginile benzii de mijloc, Imagine reprezentand banda exterioara si punctele benzii exterioare
"""
def segment_color(frame, min_area):
    frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # 1. Aplicarea mastilor imaginii
    mask_white    = mask_segmentation(frame_hls, (hue_low, lit_low, sat_low), (255, 255, 255))
    mask_yellow   = mask_segmentation(frame_hls, (hue_low_y, lit_low_y, sat_low_y), (hue_high_y, 255, 255)) 

    # 2. Extragerea benzii exterioare de culoare alba
    outerlane, points_outerlane = outer_lane(frame, mask_white, min_area + 500)
    
    # 3. Extragerea benzii din mijloc de culoare galbena
    midlane = mid_lane(frame, mask_yellow, min_area)

    return midlane,outerlane,points_outerlane