#!/bin/bash

# SadTalker Inference Script for RunPod
# Usage: ./run_inference.sh --audio_path audio.wav [options]

set -e

# Default values
AUDIO_PATH=""
IMAGE_PATH="./sadtalker_default.jpeg"
OUTPUT_DIR="./output"
DEVICE="cuda"
ENHANCER="gfpgan"
EXPRESSION_SCALE="1.0"
PREPROCESS="full"
BATCH_SIZE="10"
SAVE_BASE64="false"
STILL_MODE="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --audio_path)
            AUDIO_PATH="$2"
            shift 2
            ;;
        --image_path)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --enhancer)
            ENHANCER="$2"
            shift 2
            ;;
        --expression_scale)
            EXPRESSION_SCALE="$2"
            shift 2
            ;;
        --preprocess)
            PREPROCESS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --save_base64)
            SAVE_BASE64="true"
            shift
            ;;
        --still_mode)
            STILL_MODE="true"
            shift
            ;;
        --help)
            echo "Usage: $0 --audio_path audio.wav [options]"
            echo ""
            echo "Options:"
            echo "  --audio_path PATH        Path to audio file (required)"
            echo "  --image_path PATH        Path to source image (default: ./sadtalker_default.jpeg)"
            echo "  --output_dir PATH        Output directory (default: ./output)"
            echo "  --device DEVICE          Device to use: cuda or cpu (default: cuda)"
            echo "  --enhancer METHOD        Face enhancer: gfpgan (default: gfpgan)"
            echo "  --expression_scale NUM   Expression scale factor (default: 1.0)"
            echo "  --preprocess MODE        Preprocessing mode: crop or full (default: full)"
            echo "  --batch_size NUM         Batch size for processing (default: 10)"
            echo "  --save_base64            Save output as base64"
            echo "  --still_mode             Use still mode for generation"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --audio_path input.wav"
            echo "  $0 --audio_path input.wav --image_path face.jpg --save_base64"
            echo "  $0 --audio_path input.wav --device cpu --still_mode"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for more information"
            exit 1
            ;;
    esac
done

# Check if audio_path is provided
if [[ -z "$AUDIO_PATH" ]]; then
    echo "Error: --audio_path is required"
    echo "Usage: $0 --audio_path audio.wav [options]"
    echo "Use --help for more information"
    exit 1
fi

# Check if image exists
if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Check if audio file exists
if [[ ! -f "$AUDIO_PATH" ]]; then
    echo "Error: Audio file not found: $AUDIO_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build python command
PYTHON_CMD="python inference.py --audio_path \"$AUDIO_PATH\" --image_path \"$IMAGE_PATH\" --output_dir \"$OUTPUT_DIR\" --device \"$DEVICE\" --enhancer \"$ENHANCER\" --expression_scale \"$EXPRESSION_SCALE\" --preprocess \"$PREPROCESS\" --batch_size \"$BATCH_SIZE\""

if [[ "$SAVE_BASE64" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --save_base64"
fi

if [[ "$STILL_MODE" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --still_mode"
fi

# Log the command
echo "Running SadTalker inference..."
echo "Audio: $AUDIO_PATH"
echo "Image: $IMAGE_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""
echo "Command: $PYTHON_CMD"
echo ""

# Run the command
eval $PYTHON_CMD

# Check if successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Inference completed successfully!"
    echo "Check the output directory: $OUTPUT_DIR"
else
    echo ""
    echo "❌ Inference failed!"
    exit 1
fi 