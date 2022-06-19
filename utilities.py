import cv2

def fromFrameToJPEG(frame):
  ret, jpeg = cv2.imencode(".jpg", frame)
  return jpeg.tobytes()
