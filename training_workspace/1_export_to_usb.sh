#!/bin/bash

read -p "Inserisci la lettera della chiavetta USB (es. d, e, f): " drv
drv=${drv,,}
drv_upper=${drv^^}
DEST="/mnt/$drv/YOLO_Export"
WORKSPACE="$HOME/Tesi/Computer Vision/training_workspace"

echo "=== FASE 0: MONTAGGIO FORZATO CHIAVETTA ($drv_upper:) ==="
sudo umount -l /mnt/$drv 2>/dev/null
sudo mkdir -p /mnt/$drv
sudo mount -t drvfs $drv_upper: /mnt/$drv -o metadata,uid=$(id -u),gid=$(id -g),umask=022

# Controllo critico: se non è montata fisicamente, blocca tutto
if ! mountpoint -q /mnt/$drv; then
    echo "ERRORE CRITICO: Impossibile montare l'unità $drv_upper:. Verifica che sia inserita e accessibile."
    exit 1
fi

echo -e "\n=== FASE 1: PREPARAZIONE PACCHETTI OFFLINE ==="
mkdir -p "$WORKSPACE/wheels"
if [ ! "$(ls -A "$WORKSPACE/wheels" 2>/dev/null)" ]; then
    echo "Scaricamento dei pacchetti .whl mancanti (CUDA 11.8)..."
    source "$WORKSPACE/yolo_env/bin/activate" 2>/dev/null || echo "Attenzione: Ambiente virtuale non trovato."
    pip download --no-cache-dir --dest "$WORKSPACE/wheels" --platform manylinux2014_x86_64 --python-version 310 --implementation cp --abi cp310 --only-binary=:all: ultralytics onnx
    pip download --no-cache-dir --dest "$WORKSPACE/wheels" --platform manylinux2014_x86_64 --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
else
    echo "Cartella wheels già popolata, salto il download."
fi

echo -e "\n=== FASE 2: ESPORTAZIONE SU USB ==="
mkdir -p "$DEST/project"
echo "Copia del progetto in $DEST in corso..."

# Copia solo la cartella di training, forzando l'aggiornamento
rsync -ah --info=progress2 --ignore-existing \
    --exclude="yolo_env" \
    --exclude="data/nuove_foto" \
    "$WORKSPACE/" "$DEST/project/"

echo -e "\n=== FASE 3: GENERAZIONE SCRIPT WORKSTATION (LINUX) ==="
cat << 'EOF' > "$DEST/run_on_workstation.sh"
#!/bin/bash
USB_DIR="$(pwd)"
WORK_DIR="$HOME/Desktop/YOLO_Workspace"
LOG_FILE="$USB_DIR/project/workstation_log.txt"

exec > >(tee -i "$LOG_FILE") 2>&1

echo "=== INIZIO LOG WORKSTATION - $(date) ==="
echo "Spostamento progetto su $WORK_DIR..."
mkdir -p "$WORK_DIR"
rsync -ah --info=progress2 "$USB_DIR/project/" "$WORK_DIR/"

cd "$WORK_DIR"
echo "Creazione ambiente e installazione OFFLINE..."
python3 -m venv yolo_env
source yolo_env/bin/activate
pip install --no-index --find-links="$WORK_DIR/wheels" torch torchvision torchaudio ultralytics onnx

echo "Avvio addestramento YOLO..."
python3 train_yolo.py

echo "Salvataggio risultati su chiavetta..."
mkdir -p "$USB_DIR/project/runs"
rsync -ah --info=progress2 runs/ "$USB_DIR/project/runs/"
find . -name "*.onnx" -exec cp {} "$USB_DIR/project/data/" \;

cd ~
rm -rf "$WORK_DIR"
echo "=== FINE LOG ==="
EOF
chmod +x "$DEST/run_on_workstation.sh"

echo -e "\n=== FASE 4: SMONTAGGIO CHIAVETTA ==="
sync
sudo umount /mnt/$drv
echo "Esportazione completata in sicurezza."