from os import path, listdir
from face_recognition import face_encodings
import cv2
import time
import os
import numpy as np
import logging

known_faces_dir = 'train_images'

# logging.basicConfig(filename='model.log', format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG, encoding='utf-8')
logging.basicConfig(
    filename = 'model.log',
    # encoding = 'utf-8',
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for person_name in listdir(known_faces_dir):
        person_dir = path.join(known_faces_dir, person_name)
        if path.isdir(person_dir):
            for image_name in listdir(person_dir):
                image_path = path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                known_face_encoding = face_encodings(image)
                if not known_face_encoding:
                    print("No face detected in", image_path)
                    logging.warning("No face detected in %s", image_path)
                    continue
                print('[INFO] ',person_name, "added to known faces")
                logging.info("%s added to known faces", person_name)
                face_encoding = known_face_encoding[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)
    
    known_face_encodings_text = [','.join(str(e) for e in known_face_encoding) for known_face_encoding in known_face_encodings]
    known_face_encodings_text = '\n'.join(str(e) for e in known_face_encodings_text)
    known_face_names_text = ','.join(str(e) for e in known_face_names)
    known_faces_text = known_face_encodings_text + '\n' + known_face_names_text
    print('[INFO] Writing to known_faces.txt...')
    logging.info("Writing to known_faces.txt")
    with open('known_faces.txt', 'w') as f:
        f.write(known_faces_text)
    print('[INFO] Writing to known_faces.txt done')
    logging.info("Writing to known_faces.txt done")

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
            print('[INFO] Updating dataset...')
            logging.info("Updating dataset")
            load_known_faces()
            print('[INFO] Dataset updated')
            logging.info("Dataset updated")
            size = new_size
        time.sleep(1)


if __name__ == '__main__':
    load_known_faces()
    main()
    time.sleep(3)