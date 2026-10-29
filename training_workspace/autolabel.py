import os
import numpy as np
from ultralytics import YOLO

def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workspace, "../data/weight_from_training/best.pt")

    model = YOLO(model_path)
    
    input_dir = os.path.join(
        workspace, "../data/inputs/nuove_foto_autolabel_fiducial_training"
    )
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
        results = model.predict(source=img_path, conf=0.10, verbose=False, imgsz=1024)
        
        for result in results:
            detections = result.obb if result.obb is not None else result.boxes
            if detections is None or len(detections) == 0:
                continue
            
            is_obb = result.obb is not None
            if is_obb:
                areas = []
                for detection in detections:
                    points = detection.xyxyxyxyn[0].cpu().numpy().reshape(4, 2)
                    x_coords, y_coords = points[:, 0], points[:, 1]
                    areas.append(
                        float(
                            0.5* abs(
                                np.dot(x_coords, np.roll(y_coords, 1))
                                - np.dot(y_coords, np.roll(x_coords, 1))
                            )
                        )
                    )
            else:
                areas = [
                    float(detection.xywhn[0][2] * detection.xywhn[0][3])
                    for detection in detections
                ]
            median_area = np.median(areas)
            
            # Soglia: ±50% rispetto alla dimensione mediana
            tolerance = 0.50 
            
            label_path = os.path.join(output_labels_dir, os.path.splitext(img_name)[0] + ".txt")
            valid_boxes_count = 0
            
            with open(label_path, "w") as f:
                for detection, area in zip(detections, areas):
                    # Controlla se la dimensione è conforme
                    if (1.0 - tolerance) * median_area <= area <= (1.0 + tolerance) * median_area:
                        cls = int(detection.cls[0])
                        if is_obb:
                            points = detection.xyxyxyxyn[0].cpu().numpy().reshape(-1)
                            coordinates = " ".join(f"{float(value):.6f}" for value in points)
                            f.write(f"{cls} {coordinates}\n")
                        else:
                            x, y, w, h = detection.xywhn[0]

                            f.write(
                                f"{cls} {float(x):.6f} {float(y):.6f} "
                                f"{float(w):.6f} {float(h):.6f}\n"
                            )
                        valid_boxes_count += 1
            
            print(f"{img_name}: previste {len(detections)} | salvate valide {valid_boxes_count}")

if __name__ == '__main__':
    main()