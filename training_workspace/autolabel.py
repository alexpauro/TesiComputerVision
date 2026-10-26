import os
import numpy as np
from ultralytics import YOLO

def main():
    # Ottieni il percorso assoluto della workspace per sicurezza
    workspace = os.path.abspath(os.getcwd())
    model_path = os.path.join(workspace, "runs/detect/fiducial_model/weights/best.pt")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(workspace, "weights/best.pt")

    model = YOLO(model_path)
    
    input_dir = os.path.join(workspace, "../data/nuove_foto")
    output_labels_dir = os.path.join(workspace, "runs/detect/autolabel/labels")
    os.makedirs(output_labels_dir, exist_ok=True)

    print("Inizio inferenza e rimozione outlier (filtro visivo applicato a monte in C++)...")

    # Formati supportati, inclusi TIFF
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    if not os.path.exists(input_dir):
        print(f"Errore: cartella {input_dir} non trovata.")
        return

    for img_name in sorted(os.listdir(input_dir)):
        if not img_name.lower().endswith(valid_exts):
            continue

        img_path = os.path.join(input_dir, img_name)
        
        # Esegui l'inferenza (le immagini sono già filtrate)
        results = model.predict(source=img_path, conf=0.10, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            
            # Calcola l'area di tutte le label trovate
            areas = [float(box.xywhn[0][2] * box.xywhn[0][3]) for box in boxes]
            median_area = np.median(areas)
            
            # Soglia: ±50% rispetto alla dimensione mediana
            tolerance = 0.50 
            
            label_path = os.path.join(output_labels_dir, os.path.splitext(img_name)[0] + ".txt")
            valid_boxes_count = 0
            
            with open(label_path, "w") as f:
                for box, area in zip(boxes, areas):
                    # Controlla se la dimensione è conforme
                    if (1.0 - tolerance) * median_area <= area <= (1.0 + tolerance) * median_area:
                        cls = int(box.cls[0])
                        x, y, w, h = box.xywhn[0]
                        f.write(f"{cls} {float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}\n")
                        valid_boxes_count += 1
            
            print(f"{img_name}: previste {len(boxes)} | salvate valide {valid_boxes_count}")

if __name__ == '__main__':
    main()