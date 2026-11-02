#!/bin/bash

read -p "Inserisci la lettera dell'Hard Disk (es. d, e, f): " drv
drv=${drv,,}
drv_upper=${drv^^}

BASE_DIR="/mnt/$drv/train_pcb_detection"
DEST_DRIVE="/mnt/$drv/YOLO_Workstation_Deploy"

echo "=== FASE 0: MONTAGGIO FORZATO CON METADATI ($drv_upper:) ==="
sudo umount -l /mnt/$drv 2>/dev/null
sudo mkdir -p /mnt/$drv
sudo mount -t drvfs $drv_upper: /mnt/$drv -o metadata,uid=$(id -u),gid=$(id -g),umask=022

if ! mountpoint -q /mnt/$drv; then
    echo "ERRORE CRITICO: Impossibile montare l'unità $drv_upper:. Verifica che sia collegata e accessibile."
    exit 1
fi

echo -e "\n=== FASE 1: PREPARAZIONE DIPENDENZE E PESI OFFLINE ==="
mkdir -p "$BASE_DIR/wheels"
mkdir -p "$BASE_DIR/weights"

# 1. Download tool standalone (solo se manca)
if [ ! -f "$BASE_DIR/wheels/virtualenv.pyz" ]; then
    wget -q -O "$BASE_DIR/wheels/virtualenv.pyz" https://bootstrap.pypa.io/virtualenv.pyz
fi

echo "Controllo e scaricamento pacchetti PIP (Visibile per intercettare errori)..."
# Singolo comando senza filtri restrittivi per risolvere tutte le dipendenze in un colpo solo
pip download --no-cache-dir --dest "$BASE_DIR/wheels" --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio ultralytics onnx opencv-python-headless

# 3. Download YOLO11m (solo se manca)
if [ ! -f "$BASE_DIR/weights/yolo11m.pt" ]; then
    echo "Scaricamento dei pesi iniziali YOLO11m..."
    wget --show-progress -O "$BASE_DIR/weights/yolo11m.pt" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
fi

echo "Tutti i pacchetti necessari sono ora presenti!"

echo -e "\n=== FASE 2: ASSEMBLAGGIO CARTELLA DEPLOY SULL'HARD DISK ==="
mkdir -p "$DEST_DRIVE/project"

echo "Controllo e copia Dataset..."
if [ ! -d "$DEST_DRIVE/project/dataset" ] || [ -z "$(ls -A "$DEST_DRIVE/project/dataset" 2>/dev/null)" ]; then
    rsync -ah --info=progress2 "$BASE_DIR/merged_dataset/" "$DEST_DRIVE/project/dataset/"
else
    echo "Dataset già presente, salto la copia."
fi

echo "Controllo e copia pacchetti Wheels (Offline)..."
rsync -ah --info=progress2 "$BASE_DIR/wheels/" "$DEST_DRIVE/project/wheels/"

echo "Controllo e copia pesi YOLO (Weights)..."
if [ ! -d "$DEST_DRIVE/project/weights" ] || [ -z "$(ls -A "$DEST_DRIVE/project/weights" 2>/dev/null)" ]; then
    rsync -ah --info=progress2 "$BASE_DIR/weights/" "$DEST_DRIVE/project/weights/"
else
    echo "Pesi del modello già presenti, salto la copia."
fi

cp train_yolo.py "$DEST_DRIVE/project/"

echo -e "\n=== FASE 3: GENERAZIONE SCRIPT ANTI-TRACCIA PER WORKSTATION ==="
rm -f "$DEST_DRIVE/run_on_workstation.sh"

cat << 'EOF' > "$DEST_DRIVE/run_on_workstation.sh"
#!/bin/bash
USB_DIR="$(pwd)"
WORK_DIR="$HOME/Desktop/Temp_PCB_Training"
LOG_FILE="$USB_DIR/workstation_training_log.txt"

# Ferma l'esecuzione al primo errore critico per evitare loop rotti
set -e

exec > >(tee -i "$LOG_FILE") 2>&1

cleanup() {
    echo "=== PULIZIA DI SICUREZZA IN CORSO ==="
    cd ~
    rm -rf "$WORK_DIR"
    echo "Tracce eliminate dal PC aziendale."
}
trap cleanup EXIT ERR SIGINT SIGTERM

echo "=== INIZIO LOG WORKSTATION - $(date) ==="
mkdir -p "$WORK_DIR"
rsync -ah --info=progress2 "$USB_DIR/project/" "$WORK_DIR/"
cd "$WORK_DIR"

sed -i "s|path: .*|path: $WORK_DIR/dataset|g" dataset/dataset.yaml

echo "Creazione VENV tramite tool standalone portatile..."
python3 "$WORK_DIR/wheels/virtualenv.pyz" env_offline
source env_offline/bin/activate

echo "Installazione dipendenze OFFLINE nell'ambiente virtuale pulito..."
python3 -m pip install --no-index --find-links="$WORK_DIR/wheels" torch torchvision torchaudio ultralytics onnx opencv-python-headless

echo "Avvio addestramento YOLO11m..."
python3 train_yolo.py

echo "Salvataggio risultati su Hard Disk..."
mkdir -p "$USB_DIR/results"
rsync -ah --info=progress2 runs/ "$USB_DIR/results/"
find . -name "*.onnx" -exec cp {} "$USB_DIR/results/" \;
find . -name "*.pt" -exec cp {} "$USB_DIR/results/" \;
EOF

chmod +x "$DEST_DRIVE/run_on_workstation.sh"

echo -e "\n=== FASE 4: SMONTAGGIO CHIAVETTA ==="
sync
sudo umount /mnt/$drv 2>/dev/null
echo "Operazione completata. L'Hard Disk è stato smontato in sicurezza."
echo "Sulla workstation, avvia lo script digitando: bash run_on_workstation.sh"