#!/bin/bash

# This script runs direct inference using only gyro and accel data
# to produce a trajectory prediction with improved trajectory reconstruction.

echo "Starting RoNIN direct trajectory inference with enhanced reconstruction..."

# Input files
GYRO_FILE="/scratch/hs5580/eqnio/EqNIO/input/2025-05-20_00-05-56/GyroscopeUncalibrated.csv"
ACCEL_FILE="/scratch/hs5580/eqnio/EqNIO/input/2025-05-20_00-05-56/Accelerometer.csv"
ORIENTATION_FILE="/scratch/hs5580/eqnio/EqNIO/input/2025-05-20_00-05-56/Orientation.csv"

# Output location
OUTPUT_DIR="/scratch/hs5580/eqnio/EqNIO/output/ronin_o2/direct_inference"
mkdir -p $OUTPUT_DIR

# Model parameters
WINDOW_SIZE=200
STEP_SIZE=10

# Use pretrained ResNet model
PRETRAINED_MODEL="/scratch/hs5580/eqnio/Ronin_o2/checkpoint_38.pt"
# PRETRAINED_MODEL="/scratch/hs5580/eqnio/Ronin_so2/checkpoint_111.pt"

echo "Using pretrained model: $PRETRAINED_MODEL"

# Run enhanced direct inference with proper trajectory reconstruction
python /scratch/hs5580/eqnio/EqNIO/RONIN/source/ronin_resnet.py \
    --simple_inference \
    --gyro_file "$GYRO_FILE" \
    --accel_file "$ACCEL_FILE" \
    --orientation_file "$ORIENTATION_FILE" \
    --traj_output "$OUTPUT_DIR/predicted_trajectory.npy" \
    --arch "resnet18_eq_frame_o2" \
    --model_path "$PRETRAINED_MODEL" \
    --window_size $WINDOW_SIZE \
    --step_size $STEP_SIZE \
    --enhanced_reconstruction

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Enhanced direct trajectory inference completed successfully."
    echo "Used pretrained ResNet model: $PRETRAINED_MODEL"
    echo "Predicted trajectory saved to $OUTPUT_DIR/predicted_trajectory.npy"
    echo "Enhanced reconstruction with proper time integration used."
    if [ -f "$ORIENTATION_FILE" ]; then
        echo "Ground truth orientation data processed for reference."
    fi
    echo "Comparison plot saved to $OUTPUT_DIR/predicted_trajectory.png"
else
    echo "Direct trajectory inference failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE 