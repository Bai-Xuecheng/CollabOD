import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt')
    model.predict(source='dataset/images/test',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )