from os import path, listdir
from face_recognition import face_encodings
import cv2
import time
import os
import pickle

known_faces_dir = 'train_images'



def load_known_faces():
    if not os.path.exists('known_faces.pkl'):
        print('[INFO] Creating known_faces.pkl')
        open('known_faces.pkl', 'w').close()
    known_face_encodings = []
    known_face_names = []
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
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)
    
    known_faces = (known_face_encodings, known_face_names)
    pickle.dump(known_faces, open('known_faces.pkl', 'wb'))

def main():
    size = 0
    for path, dirs, files in os.walk('train_images'):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    new_size = 0
    while True:
        new_size = 0
        try:
            for path, dirs, files in os.walk('train_images'):
                for f in files:
                    fp = os.path.join(path, f)
                    new_size += os.path.getsize(fp)
        except:
            continue
        if new_size != size:
            print('[INFO] Updating model...')
            load_known_faces()
            print('[INFO] Model updated')
            size = new_size
        time.sleep(1)


if __name__ == '__main__':
    load_known_faces()
    main()
    time.sleep(3)