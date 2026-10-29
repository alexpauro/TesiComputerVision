#!/bin/bash

# Numero di gruppi in cui dividere i file
NUM_GRUPPI=8

echo "Avvio della suddivisione casuale di tutte le immagini in $NUM_GRUPPI gruppi..."

# 1. Array ampliato con tutti i formati immagine comuni (BMP, TIFF, RAW, ecc.)
shopt -s nullglob nocaseglob
immagini=( 
    *.jpg *.jpeg *.png *.webp *.gif 
    *.bmp *.tiff *.tif 
    *.heic *.avif 
    *.raw *.cr2 *.nef *.orf *.dng *.sr2
)
shopt -u nullglob nocaseglob

totale=${#immagini[@]}

if [ "$totale" -eq 0 ]; then
    echo "Nessuna immagine trovata in questa cartella con i formati supportati!"
    exit 1
fi

echo "Trovate $totale immagini. Mescolamento in corso..."

# 2. Crea le cartelle di destinazione (da gruppo_1 a gruppo_8)
for ((i=1; i<=NUM_GRUPPI; i++)); do
    mkdir -p "gruppo_$i"
done

# 3. Mescola l'array in modo casuale usando 'shuf' (o un fallback con awk se shuf non è disponibile)
if command -v shuf &> /dev/null; then
    file_mescolati=$(printf '%s\n' "${immagini[@]}" | shuf)
else
    file_mescolati=$(printf '%s\n' "${immagini[@]}" | awk 'BEGIN{srand()}{print rand(), $0}' | sort -n | cut -d' ' -f2-)
fi

# 4. Distribuisce i file a rotazione nei gruppi
indice=0
while IFS= read -r file; do
    [ -z "$file" ] && continue
    
    # Calcola il gruppo di destinazione da 1 a 8
    gruppo=$(( (indice % NUM_GRUPPI) + 1 ))
    
    # NOTA: Usa 'mv' per spostare i file. 
    # Se preferisci copiarli lasciando gli originali, cambia 'mv' in 'cp'
    mv "$file" "gruppo_$gruppo/"
    
    indice=$((indice + 1))
done <<< "$file_mescolati"

echo -e "\nOperazione completata con successo!"
echo "Tutte le $totale immagini sono state suddivise nelle cartelle da 'gruppo_1' a 'gruppo_$NUM_GRUPPI'."