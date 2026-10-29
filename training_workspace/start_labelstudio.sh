#!/bin/bash

# Ottieni il percorso assoluto della cartella dello script.
WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$WORKSPACE_DIR/../data/inputs/nuove_foto_autolabel_fiducial_training"
IMPORT_DIR="$WORKSPACE_DIR/import_ls_from_autolabel"

# Attiva l'ambiente virtuale posizionato nella cartella principale del progetto
if [ -d "$WORKSPACE_DIR/../yolo_env" ]; then
    source "$WORKSPACE_DIR/../yolo_env/bin/activate"
elif [ -d "$WORKSPACE_DIR/yolo_env" ]; then
    source "$WORKSPACE_DIR/yolo_env/bin/activate"
else
    echo "ERRORE: Ambiente virtuale yolo_env non trovato!"
    exit 1
fi

cd "$WORKSPACE_DIR"

echo "--- PULIZIA VECCHIE PREDIZIONI ---"
rm -rf "$WORKSPACE_DIR/runs/detect/autolabel"

echo "--- AVVIO INFERENZA AUTOLABEL (YOLO) ---"
python3 autolabel.py

echo "--- PREPARAZIONE E CONTROLLO DATI PER LABEL STUDIO ---"
rm -rf "$WORKSPACE_DIR/import_ls" "$IMPORT_DIR"
mkdir -p "$IMPORT_DIR/images" "$IMPORT_DIR/labels"
rm -rf "$IMPORT_DIR/images/"* "$IMPORT_DIR/labels/"* 2>/dev/null

# Copia delle immagini già compatibili con Label Studio.
cp "$INPUT_DIR"/*.[jJ][pP][gG] "$IMPORT_DIR/images/" 2>/dev/null
cp "$INPUT_DIR"/*.[jJ][pP][eE][gG] "$IMPORT_DIR/images/" 2>/dev/null
cp "$INPUT_DIR"/*.[pP][nN][gG] "$IMPORT_DIR/images/" 2>/dev/null
cp "$INPUT_DIR"/*.[bB][mM][pP] "$IMPORT_DIR/images/" 2>/dev/null

echo "--- CONVERSIONE TIFF IN BMP MULTICORE PER LABEL STUDIO ---"
INPUT_DIR="$INPUT_DIR" OUTPUT_DIR="$IMPORT_DIR/images" python3 - <<'PY'
import multiprocessing
import os
from pathlib import Path

from PIL import Image

input_dir = Path(os.environ["INPUT_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
tiff_files = sorted(
    path for path in input_dir.iterdir()
    if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
)

if not tiff_files:
    print("Nessun file TIFF da convertire.")
    raise SystemExit(0)

def convert_tiff(path):
    output_path = output_dir / f"{path.stem}.bmp"
    try:
        with Image.open(path) as image:
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, "white")
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            else:
                image = image.convert("RGB")
            image.save(output_path, "BMP")
        return path, output_path, None
    except Exception as error:
        return path, output_path, error

worker_count = max(1, (os.cpu_count() or 2) - 1)
with multiprocessing.Pool(processes=min(worker_count, len(tiff_files))) as pool:
    results = pool.map(convert_tiff, tiff_files)

failed = []
for source_path, output_path, error in results:
    if error is None:
        print(f"Convertito: {source_path.name} -> {output_path.name}")
    else:
        failed.append((source_path, error))
        print(f"ERRORE: impossibile convertire {source_path}: {error}", flush=True)

print(f"Conversione completata usando {min(worker_count, len(tiff_files))} worker.")
if failed:
    raise SystemExit(1)
PY

# Copia delle label generate sfruttando il percorso assoluto
if [ -d "$WORKSPACE_DIR/runs/detect/autolabel/labels" ]; then
    cp "$WORKSPACE_DIR/runs/detect/autolabel/labels/"*.txt "$IMPORT_DIR/labels/" 2>/dev/null
    echo "OK: Label YOLO copiate con successo per Label Studio."
else
    echo "ATTENZIONE: La cartella delle label di YOLO non esiste ($WORKSPACE_DIR/runs/detect/autolabel/labels)!"
fi
rm -rf "$WORKSPACE_DIR/runs/detect/autolabel"

echo "--- PULIZIA PORTE E AVVIO MICRO-SERVER ---"
fuser -k 8081/tcp 2>/dev/null
pkill -f "cors_server.py" 2>/dev/null
sleep 1

export IMPORT_DIR
cat << 'SRV' > cors_server.py
import http.server, socketserver, os, functools
DIR = os.path.abspath(os.environ["IMPORT_DIR"])
class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
socketserver.TCPServer.allow_reuse_address = True
Handler = functools.partial(CORSRequestHandler, directory=DIR)
try:
    httpd = socketserver.TCPServer(("", 8081), Handler)
    httpd.serve_forever()
except Exception:
    pass
SRV

python3 cors_server.py &
CORS_PID=$!

echo "--- GENERAZIONE JSON CONFORMATO PER LABEL STUDIO ---"
python3 -c "
import os, json, random, string
from PIL import Image

def generate_id(size=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))

images_dir = os.path.abspath('$IMPORT_DIR/images')
labels_dir = os.path.abspath('$IMPORT_DIR/labels')
tasks = []

if os.path.exists(images_dir):
    for img_name in sorted(os.listdir(images_dir)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            img_url = f'http://localhost:8081/images/{img_name}'
            img_path = os.path.join(images_dir, img_name)
            
            try:
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size
            except:
                orig_w, orig_h = 1000, 1000
            
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            
            predictions = []
            if os.path.exists(label_path):
                box_list = []
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 9:
                            # Formato OBB: classe + 8 coordinate (x1 y1 x2 y2 x3 y3 x4 y4)
                            cls = int(parts[0])
                            x_coords = [float(parts[1]), float(parts[3]), float(parts[5]), float(parts[7])]
                            y_coords = [float(parts[2]), float(parts[4]), float(parts[6]), float(parts[8])]
                            
                            # Calcola il bounding box dritto (HBB) che racchiude il box ruotato
                            min_x, max_x = min(x_coords), max(x_coords)
                            min_y, max_y = min(y_coords), max(y_coords)
                            
                            x_top_left = min_x * 100
                            y_top_left = min_y * 100
                            w_pct = (max_x - min_x) * 100
                            h_pct = (max_y - min_y) * 100
                            
                            box_list.append({
                                'id': generate_id(),
                                'original_width': orig_w,
                                'original_height': orig_h,
                                'image_rotation': 0,
                                'value': {
                                    'x': x_top_left,
                                    'y': y_top_left,
                                    'width': w_pct,
                                    'height': h_pct,
                                    'rotation': 0,
                                    'rectanglelabels': ['fiducial']
                                },
                                'from_name': 'label',
                                'to_name': 'image',
                                'type': 'rectanglelabels'
                            })
                        elif len(parts) == 5:
                            # Formato Standard: classe + x y w h
                            cls, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            box_list.append({
                                'id': generate_id(),
                                'original_width': orig_w,
                                'original_height': orig_h,
                                'image_rotation': 0,
                                'value': {
                                    'x': (x - w / 2) * 100,
                                    'y': (y - h / 2) * 100,
                                    'width': w * 100,
                                    'height': h * 100,
                                    'rotation': 0,
                                    'rectanglelabels': ['fiducial']
                                },
                                'from_name': 'label',
                                'to_name': 'image',
                                'type': 'rectanglelabels'
                            })
                
                if box_list:
                    predictions.append({'result': box_list, 'score': 0.95})
            
            tasks.append({'data': {'image': img_url}, 'predictions': predictions})

with open('import_labelstudio.json', 'w') as f:
    json.dump(tasks, f, indent=2)

print('File import_labelstudio.json generato con successo.')
"

echo "--- AVVIO LABEL STUDIO ---"
URL="http://localhost:8080"
label-studio start --port 8080 --no-browser &
LS_PID=$!

cleanup() {
    echo -e "\nChiusura..."
    kill $LS_PID 2>/dev/null
    kill $CORS_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

while ! curl -s -o /dev/null "$URL"; do sleep 1; done
if which cmd.exe > /dev/null 2>&1; then cmd.exe /c start "$URL" 2>/dev/null
elif which wslview > /dev/null 2>&1; then wslview "$URL"
fi
wait $LS_PID