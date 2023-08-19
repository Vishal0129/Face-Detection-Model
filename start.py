import threading
import os
import multiprocessing

def start_server():
    print("Starting server.py")
    os.system("python server.py")
    print('Started server.py')

def start_data_loader():
    print("Starting data_loader.py")
    os.system("python data_loader.py")
    print('Started data_loader.py')

def start_notifier():
    print("Starting notifier.py")
    os.system("python notifier.py")
    print('Started notifier.py')

def start_model():
    print("Starting model.py")
    os.system("python main.py")
    print('Started model.py')

if __name__ == "__main__":
    server_process = multiprocessing.Process(target=start_server)
    server_process.start()
    data_loader_process = multiprocessing.Process(target=start_data_loader)
    data_loader_process.start()
    notifier_process = multiprocessing.Process(target=start_notifier)
    notifier_process.start()
    model_process = multiprocessing.Process(target=start_model)
    model_process.start()

    print('END')

    while True:
        continue