import os

from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")


def download_dataset():
    api_key = ROBOFLOW_API_KEY
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("tutorial-extpj").project("brain-tumor-segmentation-jteuo")
    version = project.version(1)
    dataset = version.download("yolov8-obb")
    return dataset

def load_dataset():
    return download_dataset()

def finetune_model(model: YOLO, dataset):
    data_yaml = f"{dataset.location}/data.yaml"
    
    results = model.train(
        data=data_yaml,
        epochs=20,        # start with 50, adjust as needed
        imgsz=640,
        batch=8,          # lower if you get OOM
        lr0=1e-4,         # base LR
        patience=10,      # early stopping
        device=0,         # GPU
        project="results/brain_tumor",
        name="train",
        weight_decay=1e-4,
    )

    return results

def main():
    dataset = load_dataset()
    

    # Load a pretrained COCO segmentation model
    model = YOLO("yolo11l-seg.pt") 
    results = finetune_model(model=model, dataset=dataset)

    #print(results)

def test():
    model = YOLO("results/brain_tumor/train4/weights/best.pt")
    model.predict(
        source="input/image1.png",
        conf=0.3,
        save=True,
        project="results/brain_tumor",
        name="predictions_test"
    )
if __name__ == '__main__':
    test()
