#!/bin/bash

# 1. Installa dcraw se non è già presente nel sistema
echo "Verifica e installazione di dcraw..."
sudo apt update && sudo apt install -y dcraw

# 2. Spostati nella cartella delle immagini del cellulare
# Assicurati che il percorso corrisponda alla tua struttura reale
cd data/calib_images_s24fe || { echo "Cartella non trovata!"; exit 1; }

echo "Inizio conversione da .dng a .tiff (16-bit lineare)..."

# 3. Esegui la conversione per ogni file .dng
for file in *.dng *.raw; do
    if [ -f "$file" ]; then
        echo "Conversione in corso: $file"
        # -T esporta in formato TIFF
        # -4 mantiene i dati a 16-bit lineari (massima fedeltà geometrica e luminosa)
        dcraw -T -4 "$file"
    fi
done

echo "Conversione completata con successo."