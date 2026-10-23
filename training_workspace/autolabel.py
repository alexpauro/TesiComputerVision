from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/fiducial_model/weights/best.pt")

    # conf=0.10 forza il salvataggio anche delle predizioni incerte
    model.predict(
        source="../data/nuove_foto",
        save_txt=True,
        conf=0.10,
        project="runs/detect",
        name="autolabel",
        exist_ok=True
    )

if __name__ == '__main__':
    main()
