#!/bin/bash

# This script runs direct inference using only gyro and accel data
# to produce a trajectory prediction.

echo "Starting RoNIN direct trajectory inference..."

# Input files
GYRO_FILE="/scratch/hs5580/eqnio/EqNIO/input/custom/Gyroscope.csv"
ACCEL_FILE="/scratch/hs5580/eqnio/EqNIO/input/custom/Accelerometer.csv"
ORIENTATION_FILE="/scratch/hs5580/eqnio/EqNIO/input/custom/Orientation.csv"

# Output location
OUTPUT_DIR="/scratch/hs5580/eqnio/EqNIO/output/ronin_o2/direct_inference"
mkdir -p $OUTPUT_DIR

# Model parameters
WINDOW_SIZE=200
STEP_SIZE=10

# Run direct inference
python /scratch/hs5580/eqnio/EqNIO/RONIN/source/ronin_resnet.py \
    --simple_inference \
    --gyro_file "$GYRO_FILE" \
    --accel_file "$ACCEL_FILE" \
    --orientation_file "$ORIENTATION_FILE" \
    --traj_output "$OUTPUT_DIR/predicted_trajectory.npy" \
    --arch "resnet18_eq_frame_o2" \
    --model_path "/scratch/hs5580/eqnio/EqNIO/output/ronin_o2/checkpoints/checkpoint_latest.pt" \
    --window_size $WINDOW_SIZE \
    --step_size $STEP_SIZE

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Direct trajectory inference completed successfully."
    echo "Predicted trajectory saved to $OUTPUT_DIR/predicted_trajectory.npy"
    echo "Ground truth trajectory saved to $OUTPUT_DIR/predicted_trajectory_ground_truth.npy"
    echo "Comparison plot saved to $OUTPUT_DIR/predicted_trajectory.png"
else
    echo "Direct trajectory inference failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE 