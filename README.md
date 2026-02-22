# IROS2026

## Environment

```bash
conda create -n yolo python==3.10 -y
conda activate yolo
pip install -r requirements.txt
```



## Train
```bash
python train.py --model_yaml Path/to/model.yaml --data_yaml Path/to/data.yaml --imgsz 640 --epochs 300 --batch 16
```

## Val

## Detect