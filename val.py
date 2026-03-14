import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def parse_args():
    parser = argparse.ArgumentParser(description='Validate CollabOD models and export paper-ready metrics')

    parser.add_argument('--model', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset YAML file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split for validation')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for validation')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--project', type=str, default='runs/val', help='Project directory for saving validation results')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name for saving validation results')
    parser.add_argument('--device', type=str, default='0', help='Validation device, e.g. "0", "0,1", or "cpu"')
    parser.add_argument('--iou', type=float, default=None, help='IoU threshold for validation')
    parser.add_argument('--rect', action='store_true', help='Use rectangular validation')
    parser.add_argument('--save_json', action='store_true', help='Save results to COCO-style JSON')
    parser.add_argument('--save_txt', action='store_true', help='Save predictions to TXT files')
    parser.add_argument('--save_conf', action='store_true', help='Save confidence scores in TXT results')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--output', type=str, default='paper_data.txt', help='Output filename for exported metrics')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = YOLO(args.model)

    val_kwargs = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        rect=args.rect,
        save_json=args.save_json,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        workers=args.workers,
    )

    if args.iou is not None:
        val_kwargs['iou'] = args.iou

    result = model.val(**val_kwargs)

    if model.task == 'detect':
        length = result.box.p.size
        model_names = list(result.names.values())

        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        total_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        n_l, n_p, n_g, flops = model_info(model.model)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = [
            "GFLOPs",
            "Parameters",
            "Preprocess Time / Image",
            "Inference Time / Image",
            "Postprocess Time / Image",
            "FPS (Full Pipeline)",
            "FPS (Inference Only)",
            "Model File Size"
        ]
        model_info_table.add_row([
            f'{flops:.1f}',
            f'{n_p:,}',
            f'{preprocess_time_per_image / 1000:.6f}s',
            f'{inference_time_per_image / 1000:.6f}s',
            f'{postprocess_time_per_image / 1000:.6f}s',
            f'{1000 / total_time_per_image:.2f}',
            f'{1000 / inference_time_per_image:.2f}',
            f'{get_weight_size(args.model)}MB'
        ])
        print(model_info_table)

        model_metrics_table = PrettyTable()
        model_metrics_table.title = "Model Metrics"
        model_metrics_table.field_names = [
            "Class Name",
            "Precision",
            "Recall",
            "F1-Score",
            "mAP50",
            "mAP75",
            "mAP50-95"
        ]

        for idx in range(length):
            model_metrics_table.add_row([
                model_names[idx],
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                f"{result.box.all_ap[idx, 5]:.4f}",
                f"{result.box.ap[idx]:.4f}"
            ])

        model_metrics_table.add_row([
            "all (average)",
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(result.box.f1[:length]):.4f}",
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f"{np.mean(result.box.all_ap[:length, 5]):.4f}",
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(model_metrics_table)

        output_path = result.save_dir / args.output
        with open(output_path, 'w+', errors='ignore', encoding='utf-8') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrics_table))

        print('-' * 20, f'Results have been saved to {output_path} ...', '-' * 20)