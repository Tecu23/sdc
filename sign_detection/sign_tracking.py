import math
import cv2
import numpy as np

class SignTracking:

  # Initializarea obiectului de urmarire a semnelor
  def __init__(self):

    # modul in care se afla obiectul in acel moment
    self.mode = "detection"

    # distanta maxima
    self.max_distance = 100

    # parametrii folositi pentru metoda goodFeaturesToTrack din opencv
    self.feature_params = {
      'maxCorners': 100,
      'qualityLevel': 0.3,
      'minDistance': 7,
      'blockSize': 7 }
    # parametrii folositi pentru metoda calcOpticalFlowPyrLK din opencv
    self.lk_params = {
      'winSize': (15, 15),
      'maxLevel': 2,
      'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}  
    
    # vector de 100 de culoari random 
    self.color = np.random.randint(0, 255, (100, 3))

    # centrele obiectelor clasificate
    self.known_centers = []

    # increderea ca un obiect clasificat chiar este intradevar acolo
    self.known_centers_confidence = []

    # imaginea veche in nuante de gri
    self.old_gray = 0

    # variabila pentru pastrarea colturilor
    self.corners = []
    
    # obiectul detectat
    self.tracked_class = 0
    
    # asca pentru pastrarea unui obiect dintr-o imagine
    self.mask = 0

  """
    Metoda pentru detectarea distantei dintre 2 puncte
      IN: Cele 2 puncte
      OUT: Distanta dintre aceste puncte
  """
  def distance(self,a,b):
    return math.sqrt((float(a[1]) - float(b[1]))**2 + (float(a[0])-float(b[0]))**2)

  """
    Metoda pentru potrivirea unui centru al unui obiect detectat recent la un centru deja observat
      IN: Centrul obiectului detectat
      OUT: Daca potrivirea a fost gasita si id-ul potrivirii
  """
  def match_center(self,center):
    match_found = False
    match_idx = 0
    for i in range(len(self.known_centers)):
        if self.distance(center,self.known_centers[i]) < self.max_distance:
            match_found = True
            match_idx = i
            return match_found, match_idx
    return match_found, match_idx


  # Metoda pentru resetarea clasei de urmarire
  def reset(self):
      
    self.known_centers = []
    self.known_centers_confidence = []
    self.old_gray = 0
    self.corners = []
