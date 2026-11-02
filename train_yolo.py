from ultralytics import YOLO
import os

def main():
    weights_path = 'weights/yolo11m.pt'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Errore: Pesi non trovati in {weights_path}")
        
    model = YOLO(weights_path)

    print("Inizio addestramento con YOLO11m su RTX A4000...")
    results = model.train(
        data='dataset/dataset.yaml', 
        epochs=150,        # Aumentato per permettere al modello M di convergere al meglio
        imgsz=1024,
        batch=16,          
        device=0,
        workers=8,
        cache=False,       
        project='runs/detect',
        name='pcb_model_yolo11m'
    )

    print("Esportazione in ONNX per inferenza industriale...")
    path = model.export(format='onnx', imgsz=1024)
    print(f"Modello esportato con successo in: {path}")

if __name__ == '__main__':
    main()