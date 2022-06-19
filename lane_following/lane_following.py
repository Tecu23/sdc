import cv2

import config
from color_segmentation import segment_color
from estimation import estimate_midlane
from cleaning import correct_inner_edges
from cleaning import extend_lanes
from laneinfo import lane_info


def detect_lane(img):
  
  # 1. Decuparea imaginii
  img_cropped = img[config.CropHeight_resized:, :]

  # 2. Segmentarea imaginii
  Mid_edge_ROI,OuterLane_TwoSide,OuterLane_Points = segment_color(img_cropped,config.minArea_resized)

  # 3. Estimarea benzii
  Estimated_midlane = estimate_midlane(Mid_edge_ROI, config.MaxDist_resized)

  # 4. Corectrea imaginiilor incorecte
  OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,Offset_correction, Midlane = correct_inner_edges(OuterLane_TwoSide,Estimated_midlane,OuterLane_Points)#3ms
  Estimated_midlane,OuterLane_OneSide = extend_lanes(Estimated_midlane,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)

  # 5. Calcularea informatiilor despre drum
  Distance , Curvature, output_frame = lane_info(Mid_edge_ROI,Estimated_midlane,OuterLane_OneSide,img_cropped,Offset_correction)

  return Distance, Curvature, output_frame


# if __name__ == "__main__":
#   cap = cv2.VideoCapture('v1.h264')

#   if (cap.isOpened()== False):
#     print("Error opening video stream or file")

#   while(cap.isOpened()):
#     ret, frame = cap.read()

#     if ret == True:
#       cv2.imshow("Input", frame)
#       _, _, out_frame = detect_lane(frame)
#       cv2.imshow("Output", out_frame)


#       if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#     else:
#       break

#   cap.release()

#   cv2.destroyAllWindows()






 


