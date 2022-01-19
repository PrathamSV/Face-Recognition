from cv2 import cv2
from colorama import Fore
import os
import time

# loads pre-trained face data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# stores live webcam feed as images
webcam_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# face profile
name = input('Enter Username > ')

print(Fore.RED + 'Taking samples in . . .')
print(Fore.CYAN + '3')
time.sleep(1)
print('2')
time.sleep(1)
print('1')
print(Fore.RED + 'Capturing samples . . .')

# face sample count
count = 0
# mkdir attempts
attempt = 1

while True:
    try:
        os.mkdir(rf'A{attempt}_FaceSamples_{name}')
        break
    except FileExistsError:
        attempt += 1

os.chdir(rf'A{attempt}_FaceSamples_{name}')

while True:
    # whether frame read is successful, stores feed in frame
    successful_frame_read, frame = webcam_feed.read()
    # converts frame to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detects face in grayscaled frame, outputs fc coords as list
    face_coords = trained_face_data.detectMultiScale(frame_grayscale, scaleFactor=1.04, minNeighbors=5, minSize=(50, 50))
    # draw sqr/rect around the face
    for (x, y, w, h) in face_coords:
        x2 = x + w
        y2 = y + h
        cv2.rectangle(frame, (x, y), (x2, y2), (180, 180, 10), 2)  # (img, strtcoord, endcoord, color, thick)
        count += 1  # update count
        # save the image into folder
        cv2.imwrite(f'{str(name)}_{str(count)}.jpg', frame_grayscale[y:y2, x:x2])

    # show in-screen
    cv2.imshow('Face Detector Py', frame)
    # pause execution
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
    elif count >= 100:  # breaks after 100 samples
        print(Fore.CYAN + 'Captured Samples!')
        break

webcam_feed.release()
cv2.destroyAllWindows()
