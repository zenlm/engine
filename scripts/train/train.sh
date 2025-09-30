#!/bin/bash
# Training script for MODEL_NAME using Zen Gym

set -e

# Configuration
MODEL_NAME="MODEL_NAME"
DATASET="your/dataset"
OUTPUT_DIR="./output"
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-5

# Ensure Zen Gym is available
if [ ! -d "../../gym" ]; then
    echo "Error: Zen Gym not found. Clone from https://github.com/zenlm/gym"
    exit 1
fi

cd ../../gym

# Run training
llamafactory-cli train \
    --model_name_or_path "zenlm/${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --logging_steps 10 \
    --save_steps 100 \
    --do_train

echo "✅ Training complete! Model saved to ${OUTPUT_DIR}"
