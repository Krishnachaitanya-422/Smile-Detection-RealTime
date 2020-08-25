import cv2
from random import randrange

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')


webcam=cv2.VideoCapture(0)

while True:
    sucessful_frame_read, frame = webcam.read()

    grayscale_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = face.detectMultiScale(grayscale_img)
    

    for (x,y,w,h) in face_coordinates:
        #(x,y,w,h) = face_coordinates[]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 4)

        #slicing the frame using numpy N-Dimensional array slicing
        the_face = frame[y:y+h , x:x+w]
        

        face_grayscale=cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_coordinates = smile.detectMultiScale(face_grayscale,scaleFactor= 1.7,minNeighbors=20)


        #for (x_,y_,w_,h_) in smile_coordinates:
            
        #    cv2.rectangle(the_face, (x_, y_), (x_ +w_, y_ +h_), (randrange(128,256), randrange(128,256), randrange(128,256)), 4)

        #labiling the slmiling tag
        if len(smile_coordinates)>0:
            cv2.putText(frame, 'You are Smiling!!!', (x, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))






    cv2.imshow('MY Python Face Detection', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()

print("Code completed")