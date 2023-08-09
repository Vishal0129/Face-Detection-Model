from threading import Thread
import cv2, time
import socket
from requests import get
import numpy as np
from os import listdir, path
from face_recognition import face_encodings, compare_faces, face_locations


HOST = '192.168.0.106'
PORT = 8090
FPS = 60
IMG_WIDTH = 640
IMG_HEIGHT = 480

class Camera(object):
    def __init__(self, src=0, if_stream=False):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/FPS
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.if_stream = if_stream
        if if_stream:
            self.stream = Stream()
        
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)
            
    def show_frame(self):
        if self.if_stream:
            frame_data = cv2.imencode('.jpg', self.frame)[1].tobytes()
            frame_length = len(frame_data)
            self.stream.send(frame_length.to_bytes(4, byteorder='big'))
            self.stream.send(frame_data)
        else:
            cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(0)

class NetworkCamera(object):
    def __init__(self, url, if_stream=False):
        self.url = url
        self.model = Model('train_images')
        
        self.FPS = 1/FPS
        self.FPS_MS = int(self.FPS * 1000)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.if_stream = if_stream
        if if_stream:
            self.stream = Stream()
    
    def update(self):
        while True:
            img = get(self.url)
            self.frame = cv2.imdecode(np.array(bytearray(img.content), dtype=np.uint8), -1)
            time.sleep(self.FPS)

    def show_frames(self):
        if self.if_stream:
            frame_data = cv2.imencode('.jpg', self.frame)[1].tobytes()
            frame_length = len(frame_data)
            self.stream.send(frame_length.to_bytes(4, byteorder='big'))
            self.stream.send(frame_data)
        else:
            frame = self.model.recognize_faces(self.frame)
            cv2.imshow('frame', frame)
        cv2.waitKey(self.FPS_MS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

class Model():
    def __init__(self, known_faces_dir):
        self.known_face_encodings = []
        self.known_face_names = []
        for person_name in listdir(known_faces_dir):
            person_dir = path.join(known_faces_dir, person_name)
            if path.isdir(person_dir):
                for image_name in listdir(person_dir):
                    image_path = path.join(person_dir, image_name)
                    image = cv2.imread(image_path)
                    # image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    known_face_encodings = face_encodings(image)
                    if not known_face_encodings:
                        print("No face detected in", image_path)
                        continue
                    face_encoding = known_face_encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(person_name)
    
    def recognize_faces(self, frame):
        frame_face_locations = face_locations(frame, model='hog')
        frame_face_encodings = face_encodings(frame, frame_face_locations)
        for face_encoding, face_location in zip(frame_face_encodings, frame_face_locations):
            matches = compare_faces(self.known_face_encodings, face_encoding)
            name = "Person"
            if True in matches:
                matched_index = matches.index(True)
                name = self.known_face_names[matched_index]
            else:
                return frame
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        return frame        

class Stream():
    def __init__(self, n=1):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(n)
        self.conn, self.addr = self.server.accept()

    def send(self, data):
        self.conn.send(data)

if __name__ == '__main__':
    url = 'http://192.168.0.103:8080/shot.jpg'
    # threaded_camera = Camera(src)
    network_camera = NetworkCamera(url)
    model = Model('train_images')
    while True:
        try:
            # threaded_camera.show_frame()
            network_camera.show_frames()
        except AttributeError:
            pass
