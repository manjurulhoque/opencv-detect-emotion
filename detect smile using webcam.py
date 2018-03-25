import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")


def detect(gray, frame):
    # If faces are found, it returns the positions of detected faces as Rect(x,y,w,h)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)

    return frame


# here i need to give 0 or 1 as argument
# 0 mean my laptop's web cam
# 1 mean external web cam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()  # it return two variable, we need only last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = detect(gray, frame)
    cv2.imshow("Detecting faces", frame)
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
