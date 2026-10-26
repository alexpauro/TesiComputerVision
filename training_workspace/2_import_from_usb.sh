#!/bin/bash

read -p "Inserisci la lettera della chiavetta USB (es. d, e): " drv
drv=${drv,,}
drv_upper=${drv^^}
SRC="/mnt/$drv/YOLO_Export"
WORKSPACE="$HOME/Tesi/Computer Vision/training_workspace"
DATA_DIR="$HOME/Tesi/Computer Vision/data"

echo "=== MONTAGGIO CHIAVETTA ($drv_upper:) ==="
sudo umount -l /mnt/$drv 2>/dev/null
sudo mkdir -p /mnt/$drv
sudo mount -t drvfs $drv_upper: /mnt/$drv -o metadata,uid=$(id -u),gid=$(id -g),umask=022

if [ ! -d "$SRC" ]; then
    echo "Cartella $SRC non trovata sulla chiavetta!"
    sudo umount /mnt/$drv
    exit 1
fi

echo "Importazione file di Log..."
if [ -f "$SRC/project/workstation_log.txt" ]; then
    sudo cp "$SRC/project/workstation_log.txt" "$WORKSPACE/workstation_last_run.log"
    sudo chown $(id -u):$(id -g) "$WORKSPACE/workstation_last_run.log"
    echo "Log salvato in: workstation_last_run.log"
fi

echo "Importazione dei log di addestramento completi (runs)..."
mkdir -p "$WORKSPACE/runs"
sudo rsync -ah --info=progress2 "$SRC/project/runs/" "$WORKSPACE/runs/" 2>/dev/null
sudo chown -R $(id -u):$(id -g) "$WORKSPACE/runs/" 2>/dev/null

echo "Ricerca dell'ultimo addestramento YOLO..."
# Trova l'ultima cartella creata che inizia con fiducial_model
LATEST_RUN=$(ls -td "$SRC/project/runs/detect/fiducial_model"* 2>/dev/null | head -n 1)

if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/weights" ]; then
    echo "Trovati modelli recenti in: $LATEST_RUN/weights"
    
    # Copia il file .pt in data/
    if [ -f "$LATEST_RUN/weights/best.pt" ]; then
        sudo cp -v "$LATEST_RUN/weights/best.pt" "$DATA_DIR/best.pt"
        sudo chown $(id -u):$(id -g) "$DATA_DIR/best.pt"
        
        echo "=== GENERAZIONE ONNX RETROCOMPATIBILE (OPSET 12) ==="
        # Attiva l'ambiente virtuale di YOLO
        if [ -f "$HOME/Tesi/Computer Vision/yolo_env/bin/activate" ]; then
            source "$HOME/Tesi/Computer Vision/yolo_env/bin/activate"
        fi
        
        # Esporta il modello in ONNX direttamente in bash
        python3 -c "
from ultralytics import YOLO
import os

model_path = '$DATA_DIR/best.pt'
if os.path.exists(model_path):
    print(f'Esportazione di {model_path} in formato ONNX (opset=12, simplify=True)...')
    model = YOLO(model_path)
    model.export(format='onnx', opset=12, simplify=True)
"
        # Il file best.onnx viene generato da YOLO nella stessa cartella del .pt
        sudo chown $(id -u):$(id -g) "$DATA_DIR/best.onnx" 2>/dev/null
        echo "Modello ONNX aggiornato con successo."
    fi
else
    echo "Nessuna cartella pesi trovata in $SRC/project/runs/detect/"
fi

echo "Pulizia selettiva della chiavetta..."
sudo rm -rf "$SRC/project/runs"
sudo rm -f "$SRC/project/workstation_log.txt"

echo "=== SMONTAGGIO CHIAVETTA ==="
sync
sudo umount /mnt/$drv
echo "Importazione completata con successo. Puoi rimuovere la chiavetta in sicurezza."