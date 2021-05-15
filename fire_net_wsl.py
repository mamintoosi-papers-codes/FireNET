import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/mnt/c/temp/FireDetection/FireNet/ImageAI-master'))
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/mnt/c/temp/git/ImageAI'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection


execution_path = os.getcwd()
model_path = '/mnt/c/temp/FireDetection/FireNet/FireNET-master'

def train_detection_model():
    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="fire-dataset")
    trainer.setTrainConfig(object_names_array=["fire"], batch_size=8, num_experiments=100,
                           train_from_pretrained_model="pretrained-yolov3.h5")
    # download 'pretrained-yolov3.h5' from the link below
    # https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5
    trainer.trainModel()


def detect_from_image():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(model_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()

    # image_name = "../fire-dataset/validation/images/small (2).jpg"
    # input_image = os.path.join(execution_path, image_name)
    # input_image = "/mnt/c/temp/FireDetection/FireNet/fire-dataset/train/images/small (107).jpg"
    # input_image = "/mnt/c/temp/FireDetection/FireNet/fire-dataset/validation/images/pic (27).jpg"
    # input_image = "/mnt/c/Dropbox/MATPapers/IMG_20201201_194602.jpg"
    input_image = "images/IMG_20201201_194602_10.jpg"
    # input_image = "/mnt/c/Users/Mahmood/Downloads/Test_Dataset1__Our_Own_Dataset(1)/Test_Dataset1__Our_Own_Dataset/Fire_2/firesamp29_frame25.jpg"
    # input_image = "/mnt/c/temp/FireDetection/FireNet/FireNET-master/images/fire-sunset-01_500.jpg"
    detections = detector.detectObjectsFromImage(input_image=input_image,
                                                 output_image_path=os.path.join(execution_path, "Mahdi_kebrit3.jpg"),
                                                 minimum_percentage_probability=10)

    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def detect_from_video():
    detector = CustomVideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()

    detected_video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "video1.mp4"), frames_per_second=30, output_file_path=os.path.join(execution_path, "video1-detected"), minimum_percentage_probability=40, log_progress=True )


detect_from_image()
# detect_from_video()