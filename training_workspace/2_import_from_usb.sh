#!/bin/bash

read -p "Inserisci la lettera della chiavetta USB (es. d, e): " drv
drv=${drv,,}
drv_upper=${drv^^}
SRC="/mnt/$drv/YOLO_Export"
WORKSPACE="$HOME/Tesi/Computer Vision/training_workspace"

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

echo "Importazione dei log e dei pesi di addestramento (.pt)..."
mkdir -p "$WORKSPACE/runs"
sudo rsync -ah --info=progress2 "$SRC/project/runs/" "$WORKSPACE/runs/"
sudo chown -R $(id -u):$(id -g) "$WORKSPACE/runs/"

echo "Importazione modelli ONNX..."
sudo find "$SRC/project/data/" -name "*.onnx" -exec cp -v {} ~/Tesi/Computer\ Vision/data/ \;
sudo chown $(id -u):$(id -g) ~/Tesi/Computer\ Vision/data/*.onnx 2>/dev/null

echo "Pulizia selettiva della chiavetta (la cartella wheels viene conservata rigorosamente)..."
sudo rm -rf "$SRC/project/runs"
sudo rm -f "$SRC/project/workstation_log.txt"

echo "=== SMONTAGGIO CHIAVETTA ==="
sync
sudo umount /mnt/$drv
echo "Importazione completata con successo. Puoi rimuovere la chiavetta in sicurezza."