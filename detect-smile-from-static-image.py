import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")


def detect(gray, frame):
    # If faces are found, it returns the positions of detected faces as Rect(x,y,w,h)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)

    return frame


image = cv2.imread('man.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

frame = detect(gray, image)

cv2.imshow('detect smile', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
