#!/bin/bash

read -p "Inserisci la lettera della chiavetta USB (es. d, e): " drv
drv=${drv,,}
drv_upper=${drv^^}
SRC="/mnt/$drv/YOLO_Export"
WORKSPACE="$HOME/Tesi/Computer Vision/training_workspace"
DATA_DIR="$HOME/Tesi/Computer Vision/data/weight_from_training"
RUNS_DIR="$SRC/project/runs/detect"
WORKSPACE_RUNS_DIR="$WORKSPACE/runs/detect"

echo "=== MONTAGGIO CHIAVETTA ($drv_upper:) ==="
sudo umount -l /mnt/$drv 2>/dev/null
sudo mkdir -p /mnt/$drv
sudo mount -t drvfs $drv_upper: /mnt/$drv -o metadata,uid=$(id -u),gid=$(id -g),umask=022

if [ ! -d "$SRC" ]; then
    echo "Cartella $SRC non trovata sulla chiavetta!"
    sudo umount /mnt/$drv
    exit 1
fi

echo "Importazione file TXT..."
if [ -f "$SRC/project/workstation_log.txt" ]; then
    sudo cp "$SRC/project/workstation_log.txt" "$WORKSPACE/workstation_last_run.txt"
    sudo chown $(id -u):$(id -g) "$WORKSPACE/workstation_last_run.txt"
    echo "File TXT salvato in: workstation_last_run.txt"
fi

echo "Importazione dei log di addestramento completi (runs)..."
mkdir -p "$WORKSPACE/runs"
sudo rsync -ah --info=progress2 "$SRC/project/runs/" "$WORKSPACE/runs/" 2>/dev/null
sudo chown -R $(id -u):$(id -g) "$WORKSPACE/runs/" 2>/dev/null

echo "Ricerca dell'ultimo addestramento YOLO..."
# Trova l'ultima cartella di addestramento che inizia con fiducial_model.
LATEST_RUN=$(find "$RUNS_DIR" -maxdepth 1 -mindepth 1 \
    -type d -name "fiducial_model*" -printf '%T@ %p\n' 2>/dev/null |
    sort -nr | head -n 1 | cut -d' ' -f2-)

if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/weights" ]; then
    echo "Trovati modelli recenti in: $LATEST_RUN/weights"

    # Mantieni nella workspace solo l'ultimo modello di addestramento.
    if [ -d "$WORKSPACE_RUNS_DIR" ]; then
        find "$WORKSPACE_RUNS_DIR" -maxdepth 1 -mindepth 1 \
            -type d -name "fiducial_model*" -exec rm -rf -- {} +
    fi
    mkdir -p "$WORKSPACE_RUNS_DIR"
    rsync -ah --info=progress2 "$LATEST_RUN/" "$WORKSPACE_RUNS_DIR/fiducial_model/"
    
    # Copia i modelli già esportati in data/
    if [ -f "$LATEST_RUN/weights/best.pt" ]; then
        sudo cp -v "$LATEST_RUN/weights/best.pt" "$DATA_DIR/best.pt"
        sudo chown $(id -u):$(id -g) "$DATA_DIR/best.pt"
    fi
    if [ -f "$LATEST_RUN/weights/best.onnx" ]; then
        sudo cp -v "$LATEST_RUN/weights/best.onnx" "$DATA_DIR/best.onnx"
        sudo chown $(id -u):$(id -g) "$DATA_DIR/best.onnx"
    fi
else
    echo "Nessuna cartella pesi trovata in $RUNS_DIR/"
fi

echo "Pulizia selettiva della chiavetta..."
if [ -d "$RUNS_DIR" ]; then
    find "$RUNS_DIR" -maxdepth 1 -mindepth 1 \
        -type d -name "fiducial_model*" -exec sudo rm -rf -- {} +
fi
sudo rmdir "$RUNS_DIR" 2>/dev/null || true
sudo rmdir "$SRC/project/runs" 2>/dev/null || true
sudo rm -f "$SRC/project/workstation_log.txt"

echo "=== SMONTAGGIO CHIAVETTA ==="
sync
sudo umount /mnt/$drv
echo "Importazione completata con successo. Puoi rimuovere la chiavetta in sicurezza."