from ultralytics import YOLO

def main():
    # Carichiamo il modello YOLO26 Small OBB (Oriented Bounding Boxes)
    # Ideale per rilevare fiducial marker ruotati con precisione millimetrica
    model = YOLO('yolo26s-obb.pt')

    print("Inizio addestramento con YOLO26s-OBB su RTX A4000...")
    results = model.train(
        data='data/fiducial_dataset/dataset.yaml', 
        epochs=50, 
        imgsz=1024,        # Aumentato a 1024 per catturare i dettagli minuscoli dei fiducial
        batch=16,          # Sfruttiamo i 16 GB di VRAM della RTX A4000 per un batch size più efficiente
        device=0,          # GPU NVIDIA
        workers=8,         # Ottimizzato per velocizzare il caricamento dati
        cache=True,        # Se hai RAM a sufficienza, velocizza le epoche tenendo le immagini in memoria
        project='runs/detect',
        name='fiducial_model'
    )

    print("Esportazione in ONNX...")
    # L'esportazione OBB supporta ONNX per il deployment industriale
    path = model.export(format='onnx', imgsz=1024)
    print(f"Modello esportato con successo in: {path}")

if __name__ == '__main__':
    main()