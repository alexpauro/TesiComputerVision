#!/bin/bash

# Ottieni il percorso assoluto della cartella corrente (training_workspace)
WORKSPACE_DIR="$(pwd)"

# Attiva l'ambiente virtuale posizionato nella cartella principale del progetto
if [ -d "../yolo_env" ]; then
    source ../yolo_env/bin/activate
elif [ -d "yolo_env" ]; then
    source yolo_env/bin/activate
else
    echo "ERRORE: Ambiente virtuale yolo_env non trovato!"
    exit 1
fi

echo "--- PULIZIA VECCHIE PREDIZIONI ---"
rm -rf "$WORKSPACE_DIR/runs/detect/autolabel"

echo "--- AVVIO INFERENZA AUTOLABEL (YOLO) ---"
python3 -c "
import os
from ultralytics import YOLO

workspace = '${WORKSPACE_DIR}'
model_path = os.path.join(workspace, 'runs/detect/fiducial_model/weights/best.pt')

if not os.path.exists(model_path):
    model_path = os.path.join(workspace, 'weights/best.pt')

if os.path.exists(model_path):
    print(f'Utilizzo dei pesi da: {model_path}')
    model = YOLO(model_path)
    model.predict(
        source='../data/nuove_foto',
        save_txt=True,
        conf=0.10,
        # Forza il percorso assoluto corretto
        project=os.path.join(workspace, 'runs/detect'),
        name='autolabel',
        exist_ok=True
    )
else:
    print(f'Attenzione: Nessun file di pesi .pt trovato in {model_path}. Verranno saltate le predizioni automatiche.')
"

echo "--- PREPARAZIONE E CONTROLLO DATI PER LABEL STUDIO ---"
mkdir -p import_ls/images import_ls/labels
rm -rf import_ls/images/* import_ls/labels/* 2>/dev/null

# Copia delle immagini
cp ../data/nuove_foto/*.[jJ][pP][gG] import_ls/images/ 2>/dev/null
cp ../data/nuove_foto/*.[jJ][pP][eE][gG] import_ls/images/ 2>/dev/null
cp ../data/nuove_foto/*.[pP][nN][gG] import_ls/images/ 2>/dev/null
cp ../data/nuove_foto/*.[bB][mM][pP] import_ls/images/ 2>/dev/null

# Copia delle label generate sfruttando il percorso assoluto
if [ -d "$WORKSPACE_DIR/runs/detect/autolabel/labels" ]; then
    cp "$WORKSPACE_DIR/runs/detect/autolabel/labels/"*.txt import_ls/labels/ 2>/dev/null
    echo "OK: Label YOLO copiate con successo per Label Studio."
else
    echo "ATTENZIONE: La cartella delle label di YOLO non esiste ($WORKSPACE_DIR/runs/detect/autolabel/labels)!"
fi

echo "--- PULIZIA PORTE E AVVIO MICRO-SERVER ---"
fuser -k 8081/tcp 2>/dev/null
pkill -f "cors_server.py" 2>/dev/null
sleep 1

cat << 'SRV' > cors_server.py
import http.server, socketserver, os, functools
DIR = os.path.abspath("import_ls")
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

images_dir = 'import_ls/images'
labels_dir = 'import_ls/labels'
tasks = []

if os.path.exists(images_dir):
    for img_name in sorted(os.listdir(images_dir)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
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
                        if len(parts) >= 5:
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