import cv2
import numpy as np
import math

"""
	Metoda pentru ordonarea mai multor contururi in functie de linii sau coloane
		IN: Contururile ce trebuiesc ordonate si modul in care trebuie ordonata: linii sau coloane
		OUT: Contururile ordonate sau cele originale daca nu exista
"""
def coordinate_sorting(cnts, order):

	if cnts:
		cnt = cnts[0]
		cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
		order_list = []

		# 1. Cream ordinea in care acestea ar trebui ordonate
		if order == "rows":
			order_list.append((0, 1))
		else:
			order_list.append((1, 0))
		
		# 2. Ordonam contururile in functie de ordinea pe care am stabilit-o
		ord = np.lexsort((cnt[:, order_list[0][0]], cnt[:, order_list[0][1]]))
		ordered = cnt[ord]
		return ordered
	else:
		return cnts

"""
	Metoda pentru calcularea marimii virajului
		IN: 2 puncte reprezentand extremele traiectoriei (x1,y1) si (x2,y2)
		OUT: Unghiul de virare
"""
def lane_curvature(x1, y1, x2, y2):		
	
	offset = 90

	# 1. Calcularea pantei in functie de puncte si unghiului de virare
	if x2 - x1 != 0:
			slope = (y2 - y1) / (x2 - x1)
			inclination_angle = math.atan(slope) * (180 / np.pi)
	else:
			slope = 1000
			inclination_angle = 90

	# 2. Calcularea unghiului de virare in functie de directia in care trebuie mers
	if inclination_angle != 90:
			if inclination_angle < 0:
					angle = offset + inclination_angle
			else:
					angle = inclination_angle - offset
	else:
			angle = 0

	return angle

"""
	Metoda pentru segmentarea imaginii excluzand banda de mijloc
		IN: Imagine reprezentand marginile banzii de mijloc
		OUT: Imagine excluzand banda de mijloc
"""
def non_midlane_mask(midlane_edge):
	# 1. Crearea mastii initiale nule
	midlane_mask = np.zeros((midlane_edge.shape[0], midlane_edge.shape[1], 1), dtype=np.uint8)
	
	# 2. Gasirea tuturor contururuilor din imagine
	cnts = cv2.findContours(midlane_edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
	
	# 3. Conectarea tuturor conturilor sub forma convexa Hull si desenarea acestora pe masca
	if cnts:
		hull_list = []
		cnts = np.concatenate(cnts)
		hull = cv2.convexHull(cnts)
		hull_list.append(hull)

		midlane_mask = cv2.drawContours(midlane_mask, hull_list, 0, 255,-1)

	# 4. Aplicarea unei operatii de NOT pentru a crea imaginea de output
	output_midlane = cv2.bitwise_not(midlane_mask)

	return output_midlane

"""
	Metoda pentru calcularea punctelor drumului
		IN: Imaginile cu banda de mijloc si banda exterioara si offsetul
		OUT: Punctele drumului calculate
"""
def lane_points(midlane, outerlane, offset):

	# 1. Gasirea contururilor celor 2 benzi
	midlane_contours = cv2.findContours(midlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	outerlane_contours = cv2.findContours(outerlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

	if midlane_contours and outerlane_contours:
		
		# 2. Ordonarea contururilor in functie de randuri
		midlane_contours_sorted = coordinate_sorting(midlane_contours, "rows")
		outerlane_contours_sorted = coordinate_sorting(outerlane_contours, "rows")

		midlane_rows = midlane_contours_sorted.shape[0]
		outerlane_rows = outerlane_contours_sorted.shape[0]

		# 3. Calcularea punctelor inferioare si superioare ale benzilor exterioare si de mijloc
		midlane_low_point = midlane_contours_sorted[midlane_rows - 1, :]
		midlane_high_point = midlane_contours_sorted[0, :]

		outerlane_low_point = outerlane_contours_sorted[outerlane_rows - 1, :]
		outerlane_high_point = outerlane_contours_sorted[0, :]

		# 4. Calcularea drumului inferioare si superioare
		lane_point_lower = int((midlane_low_point[0] + outerlane_low_point[0])   / 2) + offset, int((midlane_low_point[1]  + outerlane_low_point[1] ) / 2)
		lane_point_top   = int((midlane_high_point[0] + outerlane_high_point[0]) / 2) + offset, int((midlane_high_point[1] + outerlane_high_point[1]) / 2)

		return lane_point_lower,lane_point_top
	else:
		return (0,0),(0,0)

"""
	Metoda pentru gasirea datelor despre drum
		IN: Imagine reprezentand marginile benzii de mijloc, Imagini cu banda de mijloc si banda exterioara, Imaginea initiala si offsetul calculat
		OUT: Distanta drumului, unghiul de virare si imaginea de output 
"""
def lane_info(midlane_edge, midlane, outerlane, frame, offset):

	# 1. Folosind cele 2 benzii calculam traiectoria posibila
	trajectory_low_point,trajectory_high_point = lane_points(midlane,outerlane,offset)

	# 2.Calcularea distantei si unghiului de virare din punctele din traiectoriei
	distance = -1000
	if trajectory_low_point != (0,0):
		distance = trajectory_low_point[0] - int(midlane.shape[1] / 2)

	curvature = lane_curvature(trajectory_low_point[0], trajectory_low_point[1], trajectory_high_point[0], trajectory_high_point[1])

	# 3. Pastram marginile ce fac parted din banda de mijloc
	midlane_edge = cv2.bitwise_and(midlane_edge,midlane)

	# 4. Combinam cele 2 benzi
	lanes = cv2.bitwise_or(outerlane,midlane)

	lane_projected = np.zeros(lanes.shape, lanes.dtype)
	cnts = cv2.findContours(lanes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

	# 5. Completam imaginea folosind convexPoly
	if cnts:
		cnts = np.concatenate(cnts)
		cnts = np.array(cnts)
		cv2.fillConvexPoly(lane_projected, cnts, 255)

	# 6. Cream o masca pentru extragerea benzii de mijloc
	midlane_mask = non_midlane_mask(midlane_edge)

	# 7. Extragem banda de mijloc din imagine
	lane_projected = cv2.bitwise_and(midlane_mask,lane_projected)

	lane_frame = frame

	# 8. Desenam banda proiectata pe imagine
	lane_frame[lane_projected == 255] = lane_frame[lane_projected == 255] + (0,100,0)
	lane_frame[outerlane == 255] = lane_frame[outerlane == 255] + (0,0,100)
	lane_frame[midlane == 255] = lane_frame[midlane == 255] + (100,0,0)

	output_frame = lane_frame

	# 9. Desenam directia masinii si directia drumului
	cv2.line(output_frame, (int(output_frame.shape[1] / 2), output_frame.shape[0]), (int(output_frame.shape[1] / 2), output_frame.shape[0] - int(output_frame.shape[0] / 5)), (0, 0, 255), 2)
	cv2.line(output_frame, trajectory_low_point, trajectory_high_point, (255, 0, 0), 2)

	if trajectory_low_point != (0,0):
		cv2.line(output_frame,trajectory_low_point,(int(output_frame.shape[1]/2),trajectory_low_point[1]),(255,255,0),2)# distance of car center with lane path
	
	# 10. Desenam distanta si unghiul de virare pe ecran
	text_curvature="Unghiul de virare = " + f"{curvature:.2f}"
	text_distance="Distanta = " + str(distance)
	text_size_ratio = 0.5
	cv2.putText(output_frame,text_curvature,(10,30),cv2.FONT_HERSHEY_DUPLEX,text_size_ratio,(0,255,255),1)
	cv2.putText(output_frame,text_distance,(10,50),cv2.FONT_HERSHEY_DUPLEX,text_size_ratio,(0,255,255),1)

	return distance, curvature, output_frame