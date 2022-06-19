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