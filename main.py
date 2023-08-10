from datetime import datetime
from threading import Thread
import cv2, time
import socket
from requests import get
import numpy as np
from os import listdir, path
from face_recognition import face_encodings, compare_faces, face_locations, face_distance
from queue import Queue
import cx_Oracle

HOST = '192.168.0.106'
PORT = 8090
FPS = 60
IMG_WIDTH = 640
IMG_HEIGHT = 480
DATE_FORMAT = "%Y%m%d_%H%M%S_%f"

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

    def recognize_faces(self, frame, distance_threshold=0.6, encounter_time=None):
        confidence_levels = []
        encounter_details = dict()

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
                encounter_details[name] = {'confidence': confidence, 'encounter_time': encounter_time}
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        if confidence_levels:
            # average_confidence = sum(confidence_levels) / len(confidence_levels)
            return True, frame, encounter_details
        else:
            return False, frame, encounter_details


class Stream():
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        print('[INFO] Stream server started at', HOST, PORT)
        self.conn, self.addr = self.server.accept()
        print('[INFO] Stream client connected at', self.addr)

    def send(self, data):
        self.conn.send(data)

def stream_video():
    pass

class Camera(object):
    def __init__(self, src=0, if_stream=False, camera_buffer:Queue=None, stream:Stream=None):
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
            self.stream = stream
        
    def update(self, camera_buffer:Queue=None):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                if camera_buffer:
                    # print('[INFO] Camera buffer ',camera_buffer.qsize())
                    # print('[INFO] Putting frame in camera buffer: ',camera_buffer)
                    camera_buffer.put((datetime.now().strftime(DATE_FORMAT)[:-3], self.frame))
                    camera_buffer.task_done()
            # time.sleep(self.FPS)
            # time.sleep(self.FPS)
            
    def show_frames(self):
        while True:
            try:
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
            except:
                pass

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
    def __init__(self, url, if_stream=False, camera_buffer:Queue=None, stream:Stream=None):
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
            self.stream = stream
    
    def update(self, camera_buffer:Queue=None):
        while True:
            img = get(self.url)
            self.frame = cv2.imdecode(np.array(bytearray(img.content), dtype=np.uint8), -1)
            # print('[MSG] Frame captured from network camera')
            if camera_buffer:
                # print('[INFO] Camera buffer ',camera_buffer.qsize())
                # print('[INFO] Putting frame in camera buffer: ',camera_buffer)
                camera_buffer.put((datetime.now().strftime(DATE_FORMAT)[:-3], self.frame))
                camera_buffer.task_done()
            # time.sleep(self.FPS)

    def show_frames(self):
        while True:
            try:
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
            except:
                pass

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

class DataBase():
    def __init__(self):
        self.username = "aimlb4"
        self.password = "vishalsai"
        self.host = "localhost"
        self.port = "1521"
        self.service_name = "xe"
        try:
            dsn = cx_Oracle.makedsn(self.host, self.port, service_name=self.service_name)
            self.connection = cx_Oracle.connect(self.username, self.password, dsn)
        except:
            print('[ERROR] Database connection failed')
            raise Exception('Database connection failed')
        
    def insert(self, file_name, encounter_details):
        self.cursor = self.connection.cursor()   
        for encounter in encounter_details:
            query = 'insert into encounters values(:name, :confidence, :timestamp, :image)'
            values = {
                "name": encounter,
                "confidence": '%.2f%%'%(encounter_details[encounter]["confidence"]*100),
                "timestamp": datetime.strptime(encounter_details[encounter]["encounter_time"], DATE_FORMAT),
                "image": file_name
            }
            if values["confidence"] < '43%':
                continue
            values["timestamp"] = values["timestamp"].strftime('%d-%b-%Y %H:%M:%S')
            self.cursor.execute(query, values)
        # print(values)
        self.connection.commit()
        self.cursor.close()

class Threaded_Model():
    def __init__(self, cameras: dict, if_streaming:list, streams:list[Stream]=None):
        print('[INFO] Loading model...')
        self.model = Model('train_images')
        self.models = [self.model for _ in range(len(cameras))]
        print('[INFO] Model loaded')

        if not streams:
            streams = [False for _ in range(len(cameras))]
        self.n = len(cameras)
        self.camera_names = list(cameras.keys())

        print('[INFO] Initializing camera buffers...')
        self.camera_buffers = [Queue() for _ in range(self.n)]

        print('[INFO] Initializing processed buffers...')
        self.processed_buffers = [Queue() for _ in range(self.n)]

        self.processed_frames = Queue()

        print('[INFO] Starting frame capturing...')
        self.cameras = [Network_Camera(url=cameras[self.camera_names[i]], camera_buffer=self.camera_buffers[i], if_stream=if_streaming[i], stream=streams[i]) if self.camera_names[i] == 'Network' else Camera(src=cameras[self.camera_names[i]], camera_buffer=self.camera_buffers[i], if_stream=if_streaming[i], stream=streams[i]) for i in range(self.n)]

        print('[INFO] Starting frame processing...')
        self.frame_process_threads = [Thread(target=self.process_frames, args=(i,)) for i in range(self.n)]
        for thread in self.frame_process_threads:
            thread.daemon = True
            thread.start()

        print('[INFO] Starting frame showing...')
        self.show_frames_thread = Thread(target=self.save_frames)
        self.show_frames_thread.daemon = True
        self.show_frames_thread.start()

        print('[INFO] Connecting to database...')
        self.db = DataBase()
        print('[INFO] Connected to database')

    def process_frames(self, camera_index:int):
        model = self.models[camera_index]
        while True:
            try:
                encounter_time, frame = self.camera_buffers[camera_index].get(block=False)
                found, frame, encounter_details = model.recognize_faces(frame, encounter_time=encounter_time)
                if found:
                    self.processed_buffers[camera_index].put((encounter_time, frame, encounter_details))
                self.camera_buffers[camera_index].task_done()
            except:
                continue

    def save_frames(self):
        while True:
            for i in range(self.n):
                try:
                    encounter_time, frame, encounter_details = self.processed_buffers[i].get(block=False)
                    self.processed_buffers[i].task_done()
                    # cv2.imshow('Processed Camera ' + self.camera_names[i], frame)
                    # cv2.waitKey(1)
                    file_name = self.camera_names[i] + '_' + str(encounter_time)
                    cv2.imwrite('encounters/images/' + file_name + '.jpg', frame)
                    Thread(target=self.db.insert, args=(file_name, encounter_details)).start()
                    # self.db.insert(file_name, encounter_details)
                except:
                    pass

if __name__ == '__main__':
    url = 'http://192.168.0.102:8080/shot.jpg'
    url2 = 'http://192.168.0.101:8080/shot.jpg'
    cams = {'Network': url}
    if_streaming = [True, False]
    streams = [Stream() if i else None for i in if_streaming]
    tc = Threaded_Model(cams, if_streaming=if_streaming, streams=streams)
    cameras = tc.cameras
    stream_thread = Thread(target=cameras[0].show_frames, args=())
    stream_thread.daemon = True
    stream_thread.start()
    while True:
        pass