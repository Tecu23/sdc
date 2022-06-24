import cv2

HLS=0
src=0
Hue_Low = 86
Lit_Low = 87
Sat_Low = 30#61

Hue_Low_Y = 29#30
Hue_High_Y = 58#40
Lit_Low_Y = 71#63
Sat_Low_Y = 43#81

def OnHueLowChange(val):
    global Hue_Low
    Hue_Low = val
    MaskExtract()
def OnLitLowChange(val):
    global Lit_Low
    Lit_Low = val
    MaskExtract()
def OnSatLowChange(val):
    global Sat_Low
    Sat_Low = val
    MaskExtract()

def OnHueLowChange_Y(val):
    global Hue_Low_Y
    Hue_Low_Y = val
    MaskExtract()
def OnHueHighChange_Y(val):
    global Hue_High_Y
    Hue_High_Y = val
    MaskExtract()	
def OnLitLowChange_Y(val):
    global Lit_Low_Y
    Lit_Low_Y = val
    MaskExtract()
def OnSatLowChange_Y(val):
    global Sat_Low_Y
    Sat_Low_Y = val
    MaskExtract()

def clr_segment(HSL,lower_range,upper_range):

    mask = cv2.inRange(HSL, lower_range, upper_range)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(4,4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask

def MaskExtract():
    mask   = clr_segment(HLS,(Hue_Low  ,Lit_Low   ,Sat_Low  ),(255       ,255,255))
    mask_Y = clr_segment(HLS,(Hue_Low_Y,Lit_Low_Y ,Sat_Low_Y),(Hue_High_Y,255,255))#Combine 6ms
    mask_Y_ = mask_Y != 0
    dst_Y = src * (mask_Y_[:,:,None].astype(src.dtype))
    mask_ = mask != 0

    dst = src * (mask_[:,:,None].astype(src.dtype))
    cv2.imshow('[Segment_Colour_final] mask',dst)
    cv2.imshow('[Segment_Colour_final] mask_Y',dst_Y)

if(True):

    cv2.namedWindow("[Segment_Colour_final] mask")
    cv2.namedWindow("[Segment_Colour_final] mask_Y")

    cv2.createTrackbar("Hue_L","[Segment_Colour_final] mask",Hue_Low,255,OnHueLowChange)
    cv2.createTrackbar("Lit_L","[Segment_Colour_final] mask",Lit_Low,255,OnLitLowChange)
    cv2.createTrackbar("Sat_L","[Segment_Colour_final] mask",Sat_Low,255,OnSatLowChange)

    cv2.createTrackbar("Hue_L","[Segment_Colour_final] mask_Y",Hue_Low_Y,255,OnHueLowChange_Y)
    cv2.createTrackbar("Hue_H","[Segment_Colour_final] mask_Y",Hue_High_Y,255,OnHueHighChange_Y)
    cv2.createTrackbar("Lit_L","[Segment_Colour_final] mask_Y",Lit_Low_Y,255,OnLitLowChange_Y)
    cv2.createTrackbar("Sat_L","[Segment_Colour_final] mask_Y",Sat_Low_Y,255,OnSatLowChange_Y)



def main():
    
    global HLS,src

    frame = cv2.imread("test.jpg")

    src = frame.copy()
    
    HLS = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)#2 msc

    mask   = clr_segment(HLS,(Hue_Low  ,Lit_Low   ,Sat_Low  ),(255       ,255,255))
    mask_Y = clr_segment(HLS,(Hue_Low_Y,Lit_Low_Y ,Sat_Low_Y),(Hue_High_Y,255,255))#Combine 6ms

    cv2.imshow('[Segment_Colour_final] mask',mask)
    cv2.imshow('[Segment_Colour_final] mask_Y',mask_Y)

    cv2.waitKey(0)


if __name__ == "__main__":
  main()