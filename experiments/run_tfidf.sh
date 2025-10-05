#!/bin/bash

CONFIG="configs/tfidf.yaml"

echo "Running TF-IDF + Linear SVM baseline..."
python -m src.tfidf_baseline --config $CONFIG --fold 0

echo ""
echo "TF-IDF baseline complete!"
