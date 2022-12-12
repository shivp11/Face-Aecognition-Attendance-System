from sys import path
from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
import mysql.connector
import cv2
import numpy as np
from tkinter import messagebox
from time import strftime
from datetime import datetime

# import face_recognition
# import os, sys
# import cv2
# import numpy as np
# import math

# def face_confidence(face_distance, face_match_threshold=0.6):
#     range = (1.0 - face_match_threshold)
#     linear_val = (1.0 - face_distance) / (range * 2.0)

#     if face_distance > face_match_threshold:
#         return str(round(linear_val * 100, 2)) + '%'
#     else:
#         value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
#         return str(round(value, 2)) + '%'
    
class Face_Recognition:

    def __init__(self,root):
        self.root=root
        self.root.geometry("1366x768+0+0")
        self.root.title("Face Recognition Pannel")

        # This part is image labels setting start 
        # first header image  
        img=Image.open(r"Images_GUI\banner.jpg")
        img=img.resize((1366,130),Image.Resampling.LANCZOS)
        self.photoimg=ImageTk.PhotoImage(img)

        # set image as lable
        f_lb1 = Label(self.root,image=self.photoimg)
        f_lb1.place(x=0,y=0,width=1366,height=130)

        # backgorund image 
        bg1=Image.open(r"Images_GUI\bg2.jpg")
        bg1=bg1.resize((1366,768),Image.Resampling.LANCZOS)
        self.photobg1=ImageTk.PhotoImage(bg1)

        # set image as lable
        bg_img = Label(self.root,image=self.photobg1)
        bg_img.place(x=0,y=130,width=1366,height=768)


        #title section
        title_lb1 = Label(bg_img,text="Face Recognition",font=("verdana",30,"bold"),bg="white",fg="navyblue")
        title_lb1.place(x=0,y=0,width=1366,height=45)

        # Training button 1
        std_img_btn=Image.open(r"Images_GUI\f_det.jpg")
        std_img_btn=std_img_btn.resize((180,180),Image.Resampling.LANCZOS)
        self.std_img1=ImageTk.PhotoImage(std_img_btn)

        std_b1 = Button(bg_img,command=self.face_recog,image=self.std_img1,cursor="hand2")
        std_b1.place(x=600,y=170,width=180,height=180)

        std_b1_1 = Button(bg_img,command=self.face_recog,text="Face Recognition",cursor="hand2",font=("tahoma",15,"bold"),bg="white",fg="navyblue")
        std_b1_1.place(x=600,y=350,width=180,height=45)
    #=====================Attendance===================

    def mark_attendance(self,j,i,r,n):
        with open("attendance.csv","r+",newline="\n") as f:
            myDatalist=f.readlines()
            name_list=[]
            for line in myDatalist:
                entry=line.split((","))
                name_list.append(entry[0])

            if((i not in name_list)) and ((r not in name_list)) and ((n not in name_list)) and ((j not in name_list)):
                now=datetime.now()
                d1=now.strftime("%d/%m/%Y")
                dtString=now.strftime("%H:%M:%S")
                f.writelines(f"\n{j}, {i}, {r}, {dtString}, {d1}, Present")


    # ================face recognition==================
    def face_recog(self):
        def draw_boundray(img,classifier,scaleFactor,minNeighbors,color,text,clf):
            gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            featuers=classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)

            coord=[]
            
            for (x,y,w,h) in featuers:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                id,predict=clf.predict(gray_image[y:y+h,x:x+w])

                confidence=int((100*(1-predict/300)))

                conn = mysql.connector.connect(username='root', password='admin',host='localhost',database='face_recognizer',port=3306)
                my_cursor = conn.cursor()
                
                #Fetching Department
                my_cursor.execute("select Student_ID from student where Student_ID="+str(id))
                j=my_cursor.fetchone()
                j="+".join(j)

                #Fetching Department
                my_cursor.execute("select Department from student where Student_ID="+str(id))
                n=my_cursor.fetchone()
                n="+".join(n)

                #Fetching NAME
                my_cursor.execute("select Name from student where Student_ID="+str(id))
                r=my_cursor.fetchone()
                r="+".join(r) 
                
                #Fetching Roll No
                my_cursor.execute("select Roll_No from student where Student_ID="+str(id))
                i=my_cursor.fetchone()
                i="+".join(i)


                if confidence > 84:
                    cv2.putText(img,f"Student_ID:{j}",(x,y-105),cv2.FONT_HERSHEY_COMPLEX,0.8,(64,15,223),2)
                    cv2.putText(img,f"Roll_No:{i}",(x,y-80),cv2.FONT_HERSHEY_COMPLEX,0.8,(64,15,223),2)
                    cv2.putText(img,f"Name:{r}",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(64,15,223),2)
                    cv2.putText(img,f"Department:{n}",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(64,15,223),2)
                    self.mark_attendance(j,i,r,n)
                else:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(img,"Unknown Face",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,0),3)    

                coord=[x,y,w,y]
            
            return coord   
        

        def recognize(img,clf,faceCascade):
            coord=draw_boundray(img,faceCascade,1.1,10,(255,25,255),"Face",clf)
            return img
        
        faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.read("clf.xml")

        videoCap=cv2.VideoCapture(0)

        while True:
            ret,img=videoCap.read()
            img=recognize(img,clf,faceCascade)
            cv2.imshow("Face Recognition",img)

            if cv2.waitKey(1) == 13:
                break
        videoCap.release()
        cv2.destroyAllWindows()


    # face_locations = []
    # face_encodings = []
    # face_names = []
    # known_face_encodings = []
    # known_face_names = []
    # process_current_frame = True
    
    # def __init__(self):
    #     self.encode_faces()

    # def encode_faces(self):
    #     for image in os.listdir('data_img'):
    #         face_image = face_recognition.load_image_file(f"data_img/{image}")
    #         face_encoding = face_recognition.face_encodings(face_image)[0]

    #         self.known_face_encodings.append(face_encoding)
    #         self.known_face_names.append(image)
    #     print(self.known_face_names)

    # def run_recognition(self):
    #     video_capture = cv2.VideoCapture(0)

    #     if not video_capture.isOpened():
    #         sys.exit('Video source not found...')

    #     while True:
    #         ret, frame = video_capture.read()

    #         # Only process every other frame of video to save time
    #         if self.process_current_frame:
    #             # Resize frame of video to 1/4 size for faster face recognition processing
    #             small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    #             # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #             rgb_small_frame = small_frame[:, :, ::-1]

    #             # Find all the faces and face encodings in the current frame of video
    #             self.face_locations = face_recognition.face_locations(rgb_small_frame)
    #             self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

    #             self.face_names = []
    #             for face_encoding in self.face_encodings:
    #                 # See if the face is a match for the known face(s)
    #                 matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
    #                 name = "Unknown"
    #                 confidence = '???'

    #                 # Calculate the shortest distance to face
    #                 face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

    #                 best_match_index = np.argmin(face_distances)
    #                 if matches[best_match_index]:
    #                     name = self.known_face_names[best_match_index]
    #                     confidence = face_confidence(face_distances[best_match_index])

    #                 self.face_names.append(f'{name} ({confidence})')

    #         self.process_current_frame = not self.process_current_frame

    #         # Display the results
    #         for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
    #             # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #             top *= 4
    #             right *= 4
    #             bottom *= 4
    #             left *= 4

    #             # Create the frame with the name
    #             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    #             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #             cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    #         # Display the resulting image
    #         cv2.imshow('Face Recognition', frame)

    #         # Hit 'q' on the keyboard to quit!
    #         if cv2.waitKey(1) == ord('q'):
    #             break

    #     # Release handle to the webcam
    #     video_capture.release()
    #     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     fr = Face_Recognition()
#     fr.run_recognition()

if __name__ == "__main__":
    root=Tk()
    obj=Face_Recognition(root)
    root.mainloop()
    
    
    
    
    
    
