import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser(description='Detect objects with CollabOD models')
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--source', type=str, help='Path to the source images for detection')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for detection')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project name for saving detection results')
    parser.add_argument('--name', type=str, default='exp', help='Name for saving detection results')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save detection results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()

    model = YOLO(args.model_path)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
    )