import enum
import re
import cv2

# from lane_following.color_segmentation import segment_color
# from lane_following.estimation import estimate_midlane
# from lane_following.cleaning import correct_inner_edges
# from lane_following.cleaning import extend_lanes
# from lane_following.laneinfo import lane_info

from color_segmentation import segment_color
from estimation import estimate_midlane
from cleaning import correct_inner_edges
from cleaning import extend_lanes
from laneinfo import lane_info

import skvideo.io
import numpy as np

Testing = True

Ref_imgWidth = 1920
Ref_imgHeight = 1080

Resized_width = 320
Resized_height = 240

CropHeight = 600
CropHeight_resized = int((CropHeight / Ref_imgHeight) * Resized_height)

Frame_pixels = Ref_imgWidth * Ref_imgHeight
Resized_Framepixels = Resized_width * Resized_height

Lane_Extraction_minArea_per = 1000 / Frame_pixels
minArea_resized = int(Resized_Framepixels * Lane_Extraction_minArea_per)

BWContourOpen_speed_MaxDist_per = 400 / Ref_imgHeight
MaxDist_resized = int(Resized_height * BWContourOpen_speed_MaxDist_per)

def detect_lane(img):
  
  # 1. Decuparea imaginii
  img_cropped = img[CropHeight_resized:, :]

  # 2. Segmentarea imaginii
  Mid_edge_ROI,OuterLane_TwoSide,OuterLane_Points = segment_color(img_cropped,minArea_resized)

  # 3. Estimarea benzii
  Estimated_midlane = estimate_midlane(Mid_edge_ROI, MaxDist_resized)

  # 4. Corectrea imaginiilor incorecte
  OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,Offset_correction, Midlane = correct_inner_edges(OuterLane_TwoSide, Estimated_midlane, OuterLane_Points)#3ms
  Estimated_midlane,OuterLane_OneSide = extend_lanes(Midlane,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)

  # 5. Calcularea informatiilor despre drum
  Distance , Curvature, output_frame = lane_info(Mid_edge_ROI,Estimated_midlane,OuterLane_OneSide,img_cropped,Offset_correction)

  return Distance, Curvature, output_frame


if __name__ == "__main__":
  cap = cv2.VideoCapture('curba_ext.mp4')

  if (cap.isOpened()== False):
    print("Error opening video stream or file")

  images = []
  while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
      _, _, out_frame = detect_lane(frame)

      cv2.imshow("Output", out_frame)
      images.append(out_frame)
      height,width,layers = out_frame.shape


      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    else:
      break

  video = cv2.VideoWriter("curba_exterior.avi", 0, 1, (width,height))

  for image in images:
    video.write(image)

  cap.release()
  cv2.destroyAllWindows()
  video.release()

  # frame = cv2.imread("test.jpg")

  # dist, c, out = detect_lane(frame)

  # cv2.imshow("out",out)

  # cv2.waitKey(0)








