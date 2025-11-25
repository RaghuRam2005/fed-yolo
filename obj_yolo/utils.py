# util.py
from pathlib import Path

from ultralytics.engine.model import Model

def train(partition_id:int, model:Model, data_path:Path, local_epochs:int, lr0:float):
    train_results = model.train(
        data=data_path,
        epochs=local_epochs,
        batch=16,
        imgsz=640,
        device=0,
        
        save=False,
        cache=False,
        plots=False,
        
        project='flwr_simulation',
        name=f'client_{partition_id}_train',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',     
        resume=False,

        seed=42,
        deterministic=True,
        amp=True,
        freeze=None,
        lr0=lr0,
        
        val=True,

        # Hyperparameters 
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.1,            
    )
    return train_results.box.map

def test(partition_id:int, model:Model, data_path:Path):
    validation_results = model.val(
        data=data_path,
        
        imgsz=640,
        batch=16,
        device=0,

        plots=False,
        rect=True,

        project='flwr_simulation',
        name=f'client_{partition_id}_val',

        conf=0.001,
        iou=0.6,
        max_det=300,
    )
    return validation_results.box.map

def eval_train(partition_id:int, model:Model, data_path:Path, local_epochs:int, lr0:float):
    train_results = model.train(
        data=data_path,
        epochs=local_epochs,

        batch=16,
        imgsz=640,
        device=0,
        
        save=False,
        cache=False,
        plots=False,
        
        project='flwr_simulation',
        name=f'client_{partition_id}_eval_train',
        exist_ok=True,

        pretrained=True,

        optimizer='Adam',
        resume=False,
        
        seed=42,
        deterministic=True,
        amp=True,
        freeze=None,

        lr0=lr0,
        
        val=True,

        # Hyperparameters
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.1,
    )
    return train_results.box.map