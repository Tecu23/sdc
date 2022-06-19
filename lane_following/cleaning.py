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
			order_list.append((0,1))
		else:
			order_list.append((1,0))
		
		# 2. Ordonam contururile in functie de ordinea pe care am stabilit-o
		ord = np.lexsort((cnt[:, order_list[0][0]], cnt[:, order_list[0][1]]))
		ordered = cnt[ord]
		return ordered
	else:
		return cnts

"""
	Metoda pentru a verifica daca traiectoria masinii foloseste cumva banda exterioara de pe cealalta parte
		IN: Imagine cu banda de mijloc, contururile benzii de mijloc si contururile benzii exterioare
		OUT: Varibile bool pentru verificare, path_left ne arata daca cumva urmeaza o curba la stanga, caz in care problema initiala nu se mai aplica
"""
def path_cross_midlane(midlane, contours_midlane, contours_outerlane):

	path_left = 0
	image_path = np.zeros_like(midlane)
	midlane_copy = midlane.copy()

	# 1. Sortam contururile benzii exterioare si benzii din mijloc
	contours_midlane_sorted = coordinate_sorting(contours_midlane, "rows")
	contours_outerlane_sorted = coordinate_sorting(contours_outerlane, "rows")

	if not contours_midlane:
		print("ERROR! Nu sunt contururi in imagine cu banda de mijloc")

	midlane_rows = contours_midlane_sorted.shape[0]
	outerlane_rows = contours_outerlane_sorted.shape[0]

	# 2. Calculam punctele inferioare ale celor 2 benzi
	midlane_low_point = contours_midlane_sorted[midlane_rows - 1, :]
	outerlane_low_point = contours_outerlane_sorted[outerlane_rows - 1, :]

	# 3. Calculam punctul inferior al traiectoriei, care este defapt mijlocul punctelor inferioare ale celor 2 benzi
	trajectory_low_point = (int((midlane_low_point[0] + outerlane_low_point[0]) / 2) ,int((midlane_low_point[1] + outerlane_low_point[1]) / 2))
	
	# 4. Aplicam linia traiectoriei imaginii
	cv2.line(image_path, trajectory_low_point, (int(image_path.shape[1] / 2), image_path.shape[0]), (255,255,0), 2) # distance of car center with lane path
	cv2.line(midlane_copy, tuple(midlane_low_point), (midlane_low_point[0], midlane_copy.shape[0] - 1), (255,255,0), 2) # distance of car center with lane path

	# 5. Calculam variabilele finale
	path_left = (int(image_path.shape[1] / 2) - trajectory_low_point[0]) > 0
	if np.any(cv2.bitwise_and(image_path, midlane_copy) > 0):
		return True, path_left
	else:
		return False, path_left

"""
	Metoda pentru corectarea imaginiilor incorecte
		IN: Imaginii cu banda exterioara si cea de mijloc si punctele benzii exterioare
		OUT: Imaginii reprezentand banda exterioara si cea de mijloc modificate dupa caz, conturul benziilor de mijloc si exterioare si offsetul masinii
"""
def correct_inner_edges(outerlanes,midlane,outerlane_points):
	#  Fetching the closest outer lane to mid lane is the main goal here
	
	# Variable to correct car offset if no YellowLane is Seen in Image 
	offset = 0
	outerlane_ret = np.zeros(outerlanes.shape,outerlanes.dtype)
	
	# 1. Extragerea contururilor pentru banda de mijloc si banda exterioara
	contours_midlane = cv2.findContours(midlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	contours_outerlane = cv2.findContours(outerlanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

	# 2. Verificam daca exista sau nu o banda de mijloc
	if not contours_outerlane:
		no_outerlane = True
	else:
		no_outerlane = False

	# 3. Setam primul contur ca referinta
	ref = (0,0)
	if contours_midlane:
		ref = tuple(contours_midlane[0][0][0])

	# 4a. Cazul 1 >>> Atat banda exterioara, cat si cea de mijloc este detectata si (len(outerlane_points) == 2)
	if contours_midlane and len(outerlane_points) == 2:
		point_a = outerlane_points[0]
		point_b = outerlane_points[1]
		
		closest = 0
		# 5. Gasim cea mai apropriata banda exterioara de cea mijlocie
		if distance(point_a, ref) <= distance(point_b, ref):
			closest = 0
		elif len(contours_outerlane) > 1:
			closest = 1

		outerlane_ret = cv2.drawContours(outerlane_ret, contours_outerlane, closest, 255, 2)
		contours_outerlane_ret = [contours_outerlane[closest]]

		# 6. Verificam daca banda exterioara este cea corecta prin gasirea punctelor benzii si verificam daca traiectoria trece de banda de mijloc, daca da, stergem, altfel pastram
		is_path_crossing , is_crossing_left = path_cross_midlane(midlane,contours_midlane,contours_outerlane_ret)

		if is_path_crossing:
			outerlanes = np.zeros_like(outerlanes)
		else:
			return outerlane_ret ,contours_outerlane_ret, contours_midlane, 0, midlane

	# 4b. Cazul 2 >>> Atat banda exterioara, cat si cea de mijloc este detectata si (len(outerlane_points) != 2)
	elif contours_midlane and np.any(outerlanes > 0):

		# 5. Verificam daca banda exterioara este cea corecta prin gasirea punctelor benzii si verificam daca traiectoria trece de banda de mijloc, daca da, stergem, altfel pastram
		is_path_crossing , is_crossing_left = path_cross_midlane(midlane,contours_midlane,contours_outerlane)
		if(is_path_crossing):
			outerlanes = np.zeros_like(outerlanes)
		else:
			return outerlanes ,contours_outerlane, contours_midlane, 0, midlane		


	# 4c. Cazul 3 >>> Banda de mijloc este detectata, insa banda exterioara nu este detectata sau banda exterioara trece prin banda de mijloc. Vom crea o banda exterioara care nu apare in camera
	if contours_midlane and not np.any(outerlanes > 0):	
		
		# 5. Detectarea punctelor conturului benzii de mijloc
		contours_midlane_sorted = coordinate_sorting(contours_midlane, "rows")
		midlane_rows = contours_midlane_sorted.shape[0]
		
		midlane_low_point  = contours_midlane_sorted[midlane_rows-1,:]
		midlane_high_point = contours_midlane_sorted[0,:]
		
		midlane_low_column = midlane_low_point[0]
		
		DrawRight = False

		# 6. Verificarea cazului in care exterioara trebuie creata pe partea dreapta sau stanga
		if no_outerlane:
			if midlane_low_column < int(midlane.shape[1] / 2):
				DrawRight = True
		else:
			if is_crossing_left:
				DrawRight = True

		# 7. Modificarea offetului, daca banda trebuie desenata pe partea dreapta, atunci masina trebuie sa se miste spre dreapta, si invers
		if not DrawRight:
			low_column  = 0
			high_column = 0
			offset = -20
		else:
			low_column  = int(midlane.shape[1]) - 1
			high_column = int(midlane.shape[1]) - 1
			offset = 20

		midlane_low_point[1] = midlane.shape[0]

		point_lower =  (low_column , int(midlane_low_point[1] ))
		point_top   =  (high_column, int(midlane_high_point[1]))

		# 8. Desenarea liniei dupa punctele calculate mai sus
		outerlanes = cv2.line(outerlanes, point_lower, point_top, 255, 2)	
		
		# 9. Gasirea contururile banezii exterioare desenate
		contours_outerlane = cv2.findContours(outerlanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

		return outerlanes, contours_outerlane, contours_midlane, offset, midlane
			
	# 4d. Cazul 4 >>> Banda de mijloc nu este detectata, insa banda exterioara este detectata
	elif (contours_outerlane and ( not np.any(midlane>0) )):	

		# 5. Detectarea punctelor conturului benzii de mijloc
		contours_outerlane_sorted = coordinate_sorting(contours_outerlane, "rows")
		outerlane_rows = contours_outerlane_sorted.shape[0]
		outerlane_low_point = contours_outerlane_sorted[outerlane_rows - 1, :]
		outerlane_high_point = contours_outerlane_sorted[0, :]

		# 6. Modificarea offetului, daca banda trebuie desenata pe partea dreapta, atunci masina trebuie sa se miste spre dreapta, si invers
		low_column = 0
		high_column = 0
		offset = -20

		outerlane_low_point[1] = outerlanes.shape[0]# setting mid_trajectory_lowestPoint_Row to MaxRows of Image

		point_lower =  (low_column , int(outerlane_low_point[1]))
		point_top   =  (high_column, int(outerlane_high_point[1]))

		# 7. Desenarea liniei dupa punctele calculate mai sus
		midlane = cv2.line(midlane, point_lower, point_top, 255, 2)	
		
		# 8. Gasirea contururile banezii de mijloc desenate
		contours_midlane = cv2.findContours(midlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

		return outerlanes, contours_outerlane, contours_midlane, offset, midlane
	else:

		# 4e. Cazul 5 >>> Niciunul din cazurile de mai sus nu se aplica
		return outerlanes, contours_outerlane, contours_midlane, offset, midlane

"""
	Metoda pentru extinderea benzii mai mica, pentru a echivala cele 2 benzii
		IN: Imaginile reprezentand banda de mijloc si exterioara si contururile benzilor
		OUT: Imaginile reprezentand banda de mijloc si exterioara modificate
"""
def extend_lanes(midlane, contours_midlane, contours_outerlane, outerlane):

	# 1. Ordonarea contururile celor 2 benzi dupa linii
	if contours_midlane and contours_outerlane:		
		contours_midlane_sorted = coordinate_sorting(contours_midlane, "rows")
		contours_outerlane_sorted = coordinate_sorting(contours_outerlane, "rows")

		image = midlane.shape[0]
		
		lane_rows = contours_midlane_sorted.shape[0]
		midlane_bottom_point = contours_midlane_sorted[lane_rows - 1, :]	
		
		# 3. Conectarea benzii de mijloc de imagine print-o linie verticala
		if midlane_bottom_point[1] < image:
			midlane = cv2.line(midlane, tuple(midlane_bottom_point), (midlane_bottom_point[0],image), 255)

		ref_lane_rows = contours_outerlane_sorted.shape[0]
		outerlane_bottom_point = contours_outerlane_sorted[ref_lane_rows - 1, :]

		# 3. Conectarea benzii exterioare printr-un proces de 2 pasi
		if outerlane_bottom_point[1] < image:
			if ref_lane_rows > 20:
				shift = 20
			else:
				shift = 2
			last_10_points = contours_outerlane_sorted[ref_lane_rows-shift:ref_lane_rows-1:2,:]

			# 3a. Conectarea benzii exterioare de image prin estimarea pantei si extinderea acesteia in directie, folosing 2 pucnte
			if len(last_10_points) > 1:
				
				ref_x = last_10_points[:, 0]
				ref_y = last_10_points[:, 1]
				
				# Crearea liniei si calcularea parametrilor
				ref_parameters = np.polyfit(ref_x, ref_y, 1)
				ref_slope = ref_parameters[0]
				ref_intercept = ref_parameters[1]
				
				if ref_slope < 0:
					ref_line_point_col = 0
					ref_line_point_row = ref_intercept
				else:
					ref_line_point_col = outerlane.shape[1] - 1
					ref_line_point_row = ref_slope * ref_line_point_col + ref_intercept
				
				ref_touch_point = (ref_line_point_col,int(ref_line_point_row))
				ref_bottom_point_tup = tuple(outerlane_bottom_point)
				outerlane = cv2.line(outerlane, ref_touch_point, ref_bottom_point_tup, 255)
				
				# 3b. In cazul in care banda exterioara e tot mai mica decat imagine atunci o extindem printr-o linie vericala
				if ref_line_point_row < image:
					ref_touch_point_ref = (ref_line_point_col,image)
					outerlane = cv2.line(outerlane, ref_touch_point, ref_touch_point_ref, 255)

	return midlane,outerlane