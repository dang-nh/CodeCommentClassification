#!/bin/bash

echo "========================================="
echo "Deep Learning Training Monitor"
echo "========================================="
echo ""

if ps aux | grep -q "[p]ython dl_solution.py"; then
    echo "✅ Training is RUNNING"
    echo ""
    echo "Latest training progress:"
    echo "-----------------------------------------"
    tail -30 dl_training.log | grep -E "(Epoch|Train Loss|Val F1|Fold|Best F1)"
    echo ""
    echo "-----------------------------------------"
    echo "To view full log: tail -f dl_training.log"
else
    echo "❌ Training has COMPLETED or STOPPED"
    echo ""
    if [ -f "runs/dl_solution/results.json" ]; then
        echo "✅ Results file found!"
        echo ""
        echo "Final Results:"
        cat runs/dl_solution/results.json | grep -E "(f1_|precision_|recall_)" | head -20
    fi
fi

echo ""
echo "========================================="

