import cv2
import numpy as np


def detect_signs(frame):
  
  frame_copy = frame.copy()
  frame_out = frame.copy()

   # 1. Cvt frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  NumOfVotesForCircle = 40 #parameter 1 MinVotes needed to be classified as circle
  CannyHighthresh = 200 # High threshold value for applying canny
  mindDistanBtwnCircles = 100 # kept as sign will likely not be overlapping
  max_rad = 150 # smaller circles dont have enough votes so only maxRadius need to be controlled 

  circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,mindDistanBtwnCircles,param1=CannyHighthresh,param2=NumOfVotesForCircle,minRadius=10,maxRadius=max_rad)

  out_circles = []

  if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # 4. Check if Circles larger then minim size
    for i in circles[0,:]:
      center =(i[0],i[1])
      radius = i[2] + 5
      if (radius !=5 ):

        startP = (center[0]-radius,center[1]-radius)
        endP = (center[0]+radius,center[1]+radius)
        detected_sign = frame_copy[startP[1]:endP[1],startP[0]:endP[0]]

        if(detected_sign.shape[1] and detected_sign.shape[0]):
          # draw the outer circle
          cv2.circle(frame_out,(i[0],i[1]),i[2],(0,255,0),1)
          # draw the center of the circle
          cv2.circle(frame_out,(i[0],i[1]),2,(0,0,255),3)

    
    return frame_out, circles