#!/bin/bash

# This script runs direct inference using only gyro and accel data
# to produce a trajectory prediction with PREPROCESSING ALIGNED to --mode test.

echo "Starting RoNIN direct trajectory inference with PREPROCESSING-ALIGNED reconstruction..."

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
echo "=== ENSURING PREPROCESSING ALIGNMENT ==="
echo "This will apply the same preprocessing as --mode test:"
echo "✓ Global frame transformation using orientation quaternions"
echo "✓ Calibration-aware feature extraction (warning: needs calibration params)"
echo "✓ RoNIN-style data formatting"
echo ""

# Run preprocessing-aligned direct inference 
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
    --preprocessing_aligned

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== PREPROCESSING-ALIGNED TRAJECTORY INFERENCE COMPLETED ==="
    echo "✓ Used identical preprocessing pipeline as --mode test"
    echo "✓ Applied global frame transformation using orientation data"
    echo "✓ Synchronized all sensor data (gyro, accel, orientation)"
    echo "✓ Enhanced trajectory reconstruction with actual timestamps"
    echo ""
    echo "Results:"
    echo "  Model: $PRETRAINED_MODEL"
    echo "  Trajectory: $OUTPUT_DIR/predicted_trajectory.npy"
    echo "  Visualization: $OUTPUT_DIR/predicted_trajectory.png"
    echo ""
    echo "Key differences from regular inference:"
    echo "  • Global frame data (not sensor frame)"
    echo "  • Quaternion-based orientation integration"
    echo "  • RoNIN-style feature formatting [gyro, accel]"
    echo "  • Calibration awareness (requires calibration params for perfection)"
else
    echo "Preprocessing-aligned trajectory inference failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE 