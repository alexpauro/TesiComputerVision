import os
import shutil
import random
import json
import xml.etree.ElementTree as ET
import yaml
import logging
import multiprocessing
import re
import csv
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configurazione Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

BASE_DIR = Path("/mnt/d/train_pcb_detection")
OUTPUT_DIR = BASE_DIR / "merged_dataset"
CLASSES_LIST_FILE = BASE_DIR / "classi_trovate.txt"
MAPPING_LOG_FILE = BASE_DIR / "class_mapping_auto.json"
SPLIT_RATIO = 0.8

# Le 29 Classi Target (Misc_Unknown divisa in categorie specifiche)
TARGET_CLASSES = sorted([
    "Battery", "Buzzer", "Capacitor", "Connector", "Diode", "Display", 
    "Electrolytic Capacitor", "EMI Filter", "Ferrite Bead", "Fuse", 
    "Heatsink", "IC", "Inductor", "Jumper", "LED", "MOSFET", 
    "Oscillator", "Pads", "Pins", "Relay", "Resistor", 
    "Resistor Network", "Switch_Button", "Test Point", "Text", 
    "Transformer", "Transistor", "Unknown", "Zener Diode"
])

def get_unified_class(raw_name):
    name = str(raw_name).lower().replace('"', '').strip()
    
    if "text" in name: return "Text"
    if "electrolytic" in name: return "Electrolytic Capacitor"
    if "capacitor" in name or "cap" in name.split(): return "Capacitor"
    if "resistor network" in name or "resnet" in name: return "Resistor Network"
    if "resistor" in name or "potentiometer" in name: return "Resistor"
    if "ferrite" in name: return "Ferrite Bead"
    if "inductor" in name: return "Inductor"
    if "zener" in name: return "Zener Diode"
    if "diode" in name: return "Diode"
    if "led" in name: return "LED"
    if "mosfet" in name: return "MOSFET"
    if "transistor" in name: return "Transistor"
    if "clock" in name or "oscillator" in name: return "Oscillator"
    if "ic" in name.split() or "fpga" in name: return "IC"
    if "jumper" in name: return "Jumper"
    if "connector" in name or "port" in name: return "Connector"
    if "switch" in name or "button" in name: return "Switch_Button"
    if "test point" in name: return "Test Point"
    if "pads" in name: return "Pads"
    if "pins" in name: return "Pins"
    if "transformer" in name: return "Transformer"
    if "display" in name or "disp" in name: return "Display"
    if "fuse" in name: return "Fuse"
    if "battery" in name: return "Battery"
    if "relay" in name: return "Relay"
    if "buzzer" in name or "transducer" in name: return "Buzzer"
    if "heatsink" in name: return "Heatsink"
    if "emi filter" in name or "emi" in name: return "EMI Filter"
    if "unknown" in name or "unlabeled" in name: return "Unknown"
    
    first_word = name.split()[0] if name.split() else ""
    if re.match(r'^c\d+', first_word): return "Capacitor"
    if re.match(r'^r[nmp]\d+', first_word): return "Resistor Network"
    if re.match(r'^r\d+', first_word): return "Resistor"
    if re.match(r'^fb\d+', first_word): return "Ferrite Bead"
    if re.match(r'^l\d+', first_word): return "Inductor"
    if re.match(r'^z\d+', first_word): return "Zener Diode"
    if re.match(r'^(ld|ds|led)\d*', first_word): return "LED"
    if re.match(r'^d\d+', first_word) or re.match(r'^tvs\d+', first_word): return "Diode"
    if re.match(r'^[qt]\d+', first_word): return "Transistor"
    if re.match(r'^[xy]\d+', first_word): return "Oscillator"
    if re.match(r'^u\d+', first_word) or first_word == "u": return "IC"
    if re.match(r'^jp\d+', first_word): return "Jumper"
    if re.match(r'^(j|cn|p)\d+', first_word) and not first_word.startswith('pb'): return "Connector"
    if re.match(r'^(sw|btn|pb)\d*', first_word): return "Switch_Button"
    if re.match(r'^(tp|pp)\d+', first_word): return "Test Point"
    if re.match(r'^f\d+', first_word): return "Fuse"
    if re.match(r'^bat\d*', first_word): return "Battery"
    if re.match(r'^fil\d*', first_word): return "EMI Filter"
    if re.match(r'^hsink\d*', first_word): return "Heatsink"
    
    return "Unknown"

def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        objects.append((name, convert_bbox_to_yolo((w, h), (xmin, ymin, xmax, ymax))))
    return objects

def parse_json_supervisely(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    w = data['size']['width']
    h = data['size']['height']
    
    objects = []
    for obj in data.get('objects', []):
        name = obj.get('classTitle')
        points = obj.get('points', {}).get('exterior', [])
        if len(points) == 2:
            xmin, ymin = points[0]
            xmax, ymax = points[1]
            objects.append((name, convert_bbox_to_yolo((w, h), (xmin, ymin, xmax, ymax))))
    return objects

def parse_csv_fpic(csv_path, img_path):
    objects = []
    try:
        with Image.open(img_path) as img:
            w, h = img.width, img.height
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = list(csv.reader(f))
            if not reader: return []
            
            header = [str(x).strip().lower() for x in reader[0]]
            
            # Caso 1: VGG Image Annotator (VIA)
            if 'region_attributes' in header and 'shape_attributes' in header:
                idx_s, idx_r = header.index('shape_attributes'), header.index('region_attributes')
                for row in reader[1:]:
                    try:
                        if len(row) <= max(idx_s, idx_r): continue
                        shape, region = json.loads(row[idx_s]), json.loads(row[idx_r])
                        name = list(region.values())[0] if region else "Unknown"
                        if shape.get('name') == 'rect':
                            rx, ry, rw, rh = shape['x'], shape['y'], shape['width'], shape['height']
                            objects.append((name, convert_bbox_to_yolo((w, h), (rx, ry, rx+rw, ry+rh))))
                    except: pass
                return objects
                
            # Caso 2: Formato tabellare standard FPIC
            idx_label = -1
            for i, col in enumerate(header):
                if any(k in col for k in ['label', 'class', 'name', 'component', 'type']):
                    idx_label = i; break
            
            if idx_label != -1:
                if 'xmin' in header:
                    ix, iy, ix2, iy2 = header.index('xmin'), header.index('ymin'), header.index('xmax'), header.index('ymax')
                    for row in reader[1:]:
                        try: objects.append((row[idx_label], convert_bbox_to_yolo((w, h), (float(row[ix]), float(row[iy]), float(row[ix2]), float(row[iy2])))))
                        except: pass
                    return objects
                elif 'x' in header and 'width' in header:
                    ix, iy, iw, ih = header.index('x'), header.index('y'), header.index('width'), header.index('height')
                    for row in reader[1:]:
                        try:
                            rx, ry, rw, rh = float(row[ix]), float(row[iy]), float(row[iw]), float(row[ih])
                            objects.append((row[idx_label], convert_bbox_to_yolo((w, h), (rx, ry, rx+rw, ry+rh))))
                        except: pass
                    return objects

            # Caso 3: CSV senza intestazioni
            for row in reader:
                try:
                    if len(row) >= 5:
                        if row[0].replace('.','',1).isdigit(): 
                            objects.append((row[4], convert_bbox_to_yolo((w, h), (float(row[0]), float(row[1]), float(row[2]), float(row[3])))))
                        elif row[1].replace('.','',1).isdigit():
                            objects.append((row[0], convert_bbox_to_yolo((w, h), (float(row[1]), float(row[2]), float(row[3]), float(row[4])))))
                except: pass
    except Exception: pass
    return objects

def discover_classes():
    logging.info("Avvio scansione per la ricerca delle classi (incluse annotazioni CSV)...")
    all_classes = set()
    yolo_mappings = {}

    for yaml_file in BASE_DIR.rglob("data.yaml"):
        if OUTPUT_DIR in yaml_file.parents: continue
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                names = data['names']
                names_dict = {i: n for i, n in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}
                yolo_mappings[str(yaml_file.parent)] = names_dict
                all_classes.update(names_dict.values())

    for xml_file in BASE_DIR.rglob("*.xml"):
        tree = ET.parse(xml_file)
        for obj in tree.getroot().findall('object'):
            all_classes.add(obj.find('name').text)

    for json_file in BASE_DIR.rglob("*.json"):
        if "meta.json" in json_file.name: continue
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                for obj in data.get('objects', []):
                    all_classes.add(obj.get('classTitle'))
            except Exception: pass

    # Scansione per i dataset CSV FPIC
    for csv_file in BASE_DIR.rglob("*.csv"):
        if "smd_annotation" in csv_file.parts or "annotations" in csv_file.parts:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = list(csv.reader(f))
                    if not reader: continue
                    header = [str(x).strip().lower() for x in reader[0]]
                    if 'region_attributes' in header:
                        idx = header.index('region_attributes')
                        for row in reader[1:]:
                            try:
                                region = json.loads(row[idx])
                                if region: all_classes.add(str(list(region.values())[0]))
                            except: pass
                    else:
                        idx_label = -1
                        for i, col in enumerate(header):
                            if any(k in col for k in ['label', 'class', 'name', 'component', 'type']):
                                idx_label = i; break
                        if idx_label != -1:
                            for row in reader[1:]:
                                try: all_classes.add(row[idx_label])
                                except: pass
                        else:
                            for row in reader:
                                if len(row) >= 5:
                                    if row[0].replace('.','',1).isdigit(): all_classes.add(row[4])
                                    elif row[1].replace('.','',1).isdigit(): all_classes.add(row[0])
            except Exception: pass

    with open(CLASSES_LIST_FILE, 'w') as f:
        for c in sorted(all_classes):
            f.write(f"{c}\n")

    logging.info(f"Avvio Classificazione Automatica verso le 29 classi target...")
    unified_mapping = {}
    for c in all_classes:
        unified_mapping[c] = get_unified_class(c)

    with open(MAPPING_LOG_FILE, 'w') as f:
        json.dump(unified_mapping, f, indent=4)
        
    logging.info(f"Mappatura completata. Salvata in {MAPPING_LOG_FILE} per controllo.")
    return unified_mapping, TARGET_CLASSES, yolo_mappings

def process_single_image(args):
    img_path, split, unified_mapping, unified_names, yolo_mappings, output_dir = args
    
    xml_path = img_path.with_suffix('.xml')
    json_path = img_path.parent.parent / "ann" / f"{img_path.name}.json"
    
    txt_path = img_path.with_suffix('.txt')
    if not txt_path.exists() and 'images' in img_path.parts:
        parts = list(img_path.parts)
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        txt_path = Path(*parts).with_suffix('.txt')

    # Identificazione nativa dataset FPIC tramite CSV
    csv_fpic_path = None
    if 'pcb_image' in img_path.parts:
        parts = list(img_path.parts)
        idx = len(parts) - 1 - parts[::-1].index('pcb_image')
        parts[idx] = 'smd_annotation'
        smd_dir = Path(*parts).parent
        expected_csv1 = smd_dir / f"{img_path.stem}_smd.csv"
        expected_csv2 = smd_dir / f"{img_path.stem}.csv"
        
        if expected_csv1.exists(): csv_fpic_path = expected_csv1
        elif expected_csv2.exists(): csv_fpic_path = expected_csv2

    objects = []
    if xml_path.exists():
        objects = parse_xml(xml_path)
    elif json_path.exists():
        objects = parse_json_supervisely(json_path)
    elif csv_fpic_path:
        objects = parse_csv_fpic(csv_fpic_path, img_path)
    elif txt_path.exists():
        dataset_root = None
        for p in img_path.parents:
            if (p / 'data.yaml').exists():
                dataset_root = str(p); break
                
        if dataset_root and dataset_root in yolo_mappings:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        old_id = int(parts[0])
                        name = yolo_mappings[dataset_root].get(old_id)
                        if name:
                            bbox = tuple(map(float, parts[1:]))
                            objects.append((name, bbox))
    
    if not objects:
        return (False, "no_annotation")

    yolo_lines = []
    for name, bbox in objects:
        if name in unified_mapping:
            new_id = unified_names.index(unified_mapping[name])
            yolo_lines.append(f"{new_id} {' '.join(map(str, bbox))}\n")
    
    if yolo_lines:
        rel_path = img_path.relative_to(BASE_DIR)
        unique_name = str(rel_path).replace('/', '_').replace('\\', '_')
        
        dest_img = output_dir / 'images' / split / unique_name
        dest_lbl = output_dir / 'labels' / split / f"{Path(unique_name).stem}.txt"
        
        shutil.copyfile(img_path, dest_img)
        with open(dest_lbl, 'w') as f:
            f.writelines(yolo_lines)
        return (True, "success")
        
    return (False, "no_valid_classes")

def process_datasets(unified_mapping, unified_names, yolo_mappings):
    if OUTPUT_DIR.exists():
        logging.info("Rimozione vecchia cartella merged_dataset in corso...")
        shutil.rmtree(OUTPUT_DIR)
    
    logging.info("Creazione struttura cartelle YOLO (train/val)...")
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    logging.info("Ricerca di tutte le immagini nel file system (Skippando sub-dataset inutili)...")
    all_imgs = []
    for img_path in BASE_DIR.rglob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            
            # Filtro intelligente: Ignora ritagli e maschere del dataset FPIC (incompatibili con YOLO)
            parts_lower = [p.lower() for p in img_path.parts]
            if 'segmentation' in parts_lower or 'classification' in parts_lower or 'masks' in parts_lower:
                continue
                
            if OUTPUT_DIR not in img_path.parents:
                all_imgs.append(img_path)

    total_imgs = len(all_imgs)
    logging.info(f"Trovate {total_imgs} IMMAGINI REALI DI PCB. Inizio shuffle e split...")
    
    random.shuffle(all_imgs)
    train_idx = int(total_imgs * SPLIT_RATIO)
    
    tasks = []
    for i, img_path in enumerate(all_imgs):
        split = 'train' if i < train_idx else 'val'
        tasks.append((img_path, split, unified_mapping, unified_names, yolo_mappings, OUTPUT_DIR))

    cores = multiprocessing.cpu_count()
    logging.info(f"Avvio elaborazione MULTICORE sfruttando {cores} thread...")

    processed_count = 0
    missing_ann_count = 0
    no_valid_classes_count = 0

    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            success, reason = future.result()
            
            if success:
                processed_count += 1
            elif reason == "no_annotation":
                missing_ann_count += 1
            elif reason == "no_valid_classes":
                no_valid_classes_count += 1

            if (i + 1) % 500 == 0:
                logging.info(f"Progresso elaborazione: {i + 1}/{total_imgs} file analizzati...")

    if missing_ann_count > 0:
        logging.warning(f"Ignorate {missing_ann_count} immagini di background/senza annotazioni.")
    
    logging.info(f"Copia completata. Immagini unificate con successo: {processed_count}/{total_imgs}")

    logging.info("Generazione dataset.yaml per YOLO11m in corso...")
    yaml_content = f"path: {OUTPUT_DIR.absolute()}\ntrain: images/train\nval: images/val\nnames:\n"
    for i, name in enumerate(unified_names):
        yaml_content += f"  {i}: {name}\n"
    with open(OUTPUT_DIR / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
        
    logging.info(f"=== PROCESSO COMPLETATO! ===")
    logging.info(f"Dataset definitivo e ripulito salvato in: {OUTPUT_DIR}")

if __name__ == "__main__":
    u_mapping, u_names, yolo_maps = discover_classes()
    process_datasets(u_mapping, u_names, yolo_maps)