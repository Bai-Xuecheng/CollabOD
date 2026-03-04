from ultralytics.nn.tasks import DetectionModel
import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser(description='Check the model YAML file for CollabOD')
    parser.add_argument('--model_yaml', type=str, default='./ultralytics/cfg/models/CollabOD.yaml', help='Path to the model YAML file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    DetectionModel(args.model_yaml)