import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser(description='Train YOLO Models')
    parser.add_argument('--model_yaml', type=str, help='Path to the model YAML file')
    parser.add_argument('--data_yaml', type=str, help='Path to the data YAML file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer for training')
    parser.add_argument('--project', type=str, default='runs/train', help='Project name for saving training results')
    parser.add_argument('--name', type=str, default='exp', help='Name for saving training results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()

    model = YOLO(args.model_yaml)
    model.train(
        data=args.data_yaml,
        cache=False,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        close_mosaic=0,
        workers=0,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
    )