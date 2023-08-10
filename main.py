from datetime import datetime
from threading import Thread
import cv2, time
import socket
from requests import get
import numpy as np
from os import listdir, path
from face_recognition import face_encodings, compare_faces, face_locations, face_distance
from queue import Queue


HOST = '192.168.0.106'
PORT = 8090
FPS = 60
IMG_WIDTH = 640
IMG_HEIGHT = 480

class Model(object):
    def __init__(self, known_faces_dir):
        self.known_face_encodings = []
        self.known_face_names = []
        for person_name in listdir(known_faces_dir):
            person_dir = path.join(known_faces_dir, person_name)
            if path.isdir(person_dir):
                for image_name in listdir(person_dir):
                    image_path = path.join(person_dir, image_name)
                    image = cv2.imread(image_path)
                    known_face_encodings = face_encodings(image)
                    if not known_face_encodings:
                        print("No face detected in", image_path)
                        continue
                    print('[MSG] ',person_name, "added to known faces")
                    face_encoding = known_face_encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(person_name)
    
    # def recognize_faces(self, frame):
    #     i=0
    #     frame_face_locations = face_locations(frame, model='hog')
    #     frame_face_encodings = face_encodings(frame, frame_face_locations)
    #     for face_encoding, face_location in zip(frame_face_encodings, frame_face_locations):
    #         matches = compare_faces(self.known_face_encodings, face_encoding)
            
    #         name = "Person"
    #         confidence = 0.0

    #         if True in matches:
    #             i+=1
    #             matched_index = matches.index(True)
    #             confidence = 1.0 - face_distance(self.known_face_encodings[matched_index], face_encoding)
    #             name = self.known_face_names[matched_index]
    #         else:
    #             continue
    #         top, right, bottom, left = face_location
    #         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    #         cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
    #         cv2.putText(frame, f"{name} ({confidence:.2f})", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.5 (255, 255, 255), 1)

    #     return (True, frame) if i>0 else (False, frame)
    def recognize_faces(self, frame, distance_threshold=0.6):
        confidence_levels = []

        frame_face_locations = face_locations(frame, model='hog')
        frame_face_encodings = face_encodings(frame, frame_face_locations)

        for face_encoding, face_location in zip(frame_face_encodings, frame_face_locations):
            distances = face_distance(self.known_face_encodings, face_encoding)
            matches = [distance <= distance_threshold for distance in distances]

            if any(matches):
                matched_index = matches.index(True)
                confidence = 1.0 - distances[matched_index]
                confidence_levels.append(confidence)

                name = self.known_face_names[matched_index]
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        if confidence_levels:
            # average_confidence = sum(confidence_levels) / len(confidence_levels)
            return True, frame
        else:
            return False, frame
    
class Camera(object):
    def __init__(self, src=0, if_stream=False, camera_buffer:Queue=None):
        self.src = src
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       
        self.FPS = 1/FPS
        self.FPS_MS = int(self.FPS * 1000)
        
        self.thread = Thread(target=self.update, args=(camera_buffer,))
        self.thread.daemon = True
        print('[INFO] Frame capturing started from camera', src)
        self.thread.start()

        self.if_stream = if_stream
        if if_stream:
            self.stream = Stream()
        
    def update(self, camera_buffer:Queue=None):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                if camera_buffer:
                    # print('[INFO] Camera buffer ',camera_buffer.qsize())
                    # print('[INFO] Putting frame in camera buffer: ',camera_buffer)
                    camera_buffer.put((datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3], self.frame))
                    camera_buffer.task_done()
            # time.sleep(self.FPS)
            # time.sleep(self.FPS)
            
    def show_frames(self):
        if self.if_stream:
            frame_data = cv2.imencode('.jpg', self.frame)[1].tobytes()
            frame_length = len(frame_data)
            self.stream.send(frame_length.to_bytes(4, byteorder='big'))
            self.stream.send(frame_data)
        else:
            cv2.imshow('Camera'+str(self.src), self.frame)
        cv2.waitKey(self.FPS_MS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(0)

    def show_frames_thread(self):
        while True:
            try:
                cv2.imshow('Camera '+str(self.src), self.frame)
                cv2.waitKey(self.FPS_MS)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    exit(0)
            except:
                pass

class Network_Camera(object):
    def __init__(self, url, if_stream=False, camera_buffer:Queue=None):
        self.url = url
        self.camera_buffer = camera_buffer
        self.FPS = 1/FPS
        self.FPS_MS = int(self.FPS * 1000)

        self.thread = Thread(target=self.update, args=(camera_buffer,))
        self.thread.daemon = True
        print('[INFO] Frame capturing started from network camera', url)
        self.thread.start()
        self.if_stream = if_stream
        if if_stream:
            self.stream = Stream()
    
    def update(self, camera_buffer:Queue=None):
        while True:
            img = get(self.url)
            self.frame = cv2.imdecode(np.array(bytearray(img.content), dtype=np.uint8), -1)
            # print('[MSG] Frame captured from network camera')
            if camera_buffer:
                # print('[INFO] Camera buffer ',camera_buffer.qsize())
                # print('[INFO] Putting frame in camera buffer: ',camera_buffer)
                camera_buffer.put((datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3], self.frame))
                camera_buffer.task_done()
            # time.sleep(self.FPS)

    def show_frames(self):
        if self.if_stream:
            frame_data = cv2.imencode('.jpg', self.frame)[1].tobytes()
            frame_length = len(frame_data)
            self.stream.send(frame_length.to_bytes(4, byteorder='big'))
            self.stream.send(frame_data)
        else:
            cv2.imshow('Network Camera'+self.url, self.frame)
        cv2.waitKey(self.FPS_MS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

    def show_frames_thread(self):
        while True:
            try:
                cv2.imshow('Network Camera '+self.url, self.frame)
                cv2.waitKey(self.FPS_MS)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    exit(0)
            except:
                pass 

class Threaded_Cameras():
    def __init__(self, cameras:dict):
        self.n = len(cameras)
        self.cameras = [Network_Camera(cameras[i]) if i=='Network' else Camera(cameras[i]) for i in cameras]
        self.threads = [Thread(target=self.cameras[i].show_frames_thread, args=()) for i in range(self.n)]
        for thread in self.threads:
            thread.daemon = True
            thread.start()
    
    def get_cameras(self):
        return self.cameras

class Threaded_Model():
    def __init__(self, cameras: dict):
        print('[INFO] Loading model...')
        self.model = Model('train_images')
        self.models = [self.model for _ in range(len(cameras))]
        print('[INFO] Model loaded')

        self.n = len(cameras)
        self.camera_names = list(cameras.keys())

        print('[INFO] Initializing camera buffers...')
        self.camera_buffers = [Queue() for _ in range(self.n)]

        print('[INFO] Initializing processed buffers...')
        self.processed_buffers = [Queue() for _ in range(self.n)]

        self.processed_frames = Queue()

        print('[INFO] Starting frame capturing...')
        self.cameras = [Network_Camera(url=cameras[self.camera_names[i]], camera_buffer=self.camera_buffers[i]) if self.camera_names[i] == 'Network' else Camera(src=cameras[self.camera_names[i]], camera_buffer=self.camera_buffers[i]) for i in range(self.n)]

        print('[INFO] Starting frame processing...')
        self.frame_process_threads = [Thread(target=self.process_frames, args=(i,)) for i in range(self.n)]
        for thread in self.frame_process_threads:
            thread.daemon = True
            thread.start()

        print('[INFO] Starting frame showing...')
        self.show_frames_thread = Thread(target=self.save_frames)
        self.show_frames_thread.daemon = True
        self.show_frames_thread.start()

    def process_frames(self, camera_index:int):
        model = self.models[camera_index]
        while True:
            try:
                encounter_time, frame = self.camera_buffers[camera_index].get(block=False)
                found, frame = model.recognize_faces(frame)
                if found:
                    self.processed_buffers[camera_index].put((encounter_time, frame))
                self.camera_buffers[camera_index].task_done()
            except:
                continue

    def save_frames(self):
        while True:
            for i in range(self.n):
                try:
                    encounter_time, frame = self.processed_buffers[i].get(block=False)
                    # cv2.imshow('Processed Camera ' + self.camera_names[i], frame)
                    # cv2.waitKey(1)
                    cv2.imwrite('encounters/'+self.camera_names[i]+str(encounter_time)+'.jpg', frame)
                    self.processed_buffers[i].task_done()
                except:
                    pass

class Stream():
    def __init__(self, n=1):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(n)
        self.conn, self.addr = self.server.accept()

    def send(self, data):
        self.conn.send(data)

if __name__ == '__main__':
    url = 'http://192.168.0.102:8080/shot.jpg'
    url2 = 'http://192.168.0.101:8080/shot.jpg'
    # threaded_camera = Camera(src)
    # network_camera = Network_Camera(url)
    # network_camera = Camera()
    # model = Model('train_images')
    cameras = {'Network': url, 'Webcam': 0}
    tc = Threaded_Model(cameras)
    # camera_objects = tc.get_cameras() 
    while True:
        try:
            # print(tc.processed_frames.qsize())
            # print(tc.processed_buffers[0].qsize())
            pass
            # threaded_camera.show_frame()
            # network_camera.show_frames()
        except AttributeError:
            pass
