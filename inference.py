#!/usr/bin/env python3
"""
SadTalker Inference Script
Generate talking face video from audio and image using SadTalker
"""

import os
import sys
import argparse
import json
from time import strftime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SadTalker Inference Script')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file')
    parser.add_argument('--image_path', type=str, default='./sadtalker_default.jpeg', help='Path to the source image')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for inference')
    parser.add_argument('--enhancer', type=str, default='gfpgan', help='Face enhancer to use')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for face rendering')
    parser.add_argument('--expression_scale', type=float, default=1.0, help='Expression scale factor')
    parser.add_argument('--still_mode', action='store_true', help='Use still mode for generation')
    parser.add_argument('--preprocess', type=str, default='full', choices=['crop', 'full'], help='Preprocessing mode')
    parser.add_argument('--save_base64', action='store_true', help='Save output video as base64')
    
    return parser.parse_args()

def initialize_models(device, checkpoints_dir="./checkpoints"):
    """Initialize SadTalker models"""
    try:
        # Try to import required modules
        # Note: These imports may fail if modules are missing
        from src.utils.preprocess import CropAndExtract
        from src.test_audio2coeff import Audio2Coeff  
        from src.facerender.animate_onnx import AnimateFromCoeff
        from src.generate_batch import get_data
        from src.generate_facerender_batch import get_facerender_data
        from src.utils.init_path import init_path
        
        logger.info("Initializing SadTalker models...")
        
        # Initialize paths
        sadtalker_paths = init_path(checkpoints_dir, 
                                  os.path.join(os.getcwd(), 'src/config'), 
                                  "256", False, "full")
        
        # Initialize models
        preprocess_model = CropAndExtract(sadtalker_paths, device)
        audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
        
        logger.info("Models initialized successfully")
        return preprocess_model, audio_to_coeff, animate_from_coeff, get_data, get_facerender_data
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all SadTalker dependencies are installed and modules are available")
        return None, None, None, None, None

def run_inference(audio_path, image_path, output_dir, device, args):
    """Run SadTalker inference"""
    
    # Initialize models
    preprocess_model, audio_to_coeff, animate_from_coeff, get_data, get_facerender_data = initialize_models(device)
    
    if not all([preprocess_model, audio_to_coeff, animate_from_coeff, get_data, get_facerender_data]):
        logger.error("Model initialization failed")
        return None
    
    # Create output directory
    save_dir = os.path.join(output_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Preprocessing
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        
        logger.info("Starting preprocessing...")
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            image_path, first_frame_dir, args.preprocess, source_image_flag=True)
        
        # Audio to coefficients
        logger.info("Converting audio to coefficients...")
        ref_eyeblink_coeff_path = None
        ref_pose_coeff_path = None
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still_mode)
        coeff_path = audio_to_coeff.generate(batch, save_dir, 0, ref_pose_coeff_path)
        
        # Generate video
        logger.info("Generating video...")
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                   args.batch_size, None, None, None,
                                   expression_scale=args.expression_scale, 
                                   still_mode=args.still_mode, 
                                   preprocess=args.preprocess)
        
        video_path = animate_from_coeff.generate_deploy(data, save_dir, image_path, crop_info,
                                                       enhancer=args.enhancer, 
                                                       background_enhancer=None, 
                                                       preprocess=args.preprocess)
        
        logger.info(f"Video generated: {video_path}")
        
        # Handle output
        result = {"video_path": video_path}
        
        if args.save_base64:
            import base64
            with open(video_path, "rb") as file:
                video_data = base64.b64encode(file.read()).decode("utf-8")
            result["video_base64"] = video_data
            
            # Save base64 to file
            base64_path = os.path.join(save_dir, "video_base64.txt")
            with open(base64_path, "w") as f:
                f.write(video_data)
            logger.info(f"Base64 video saved to: {base64_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None

def main():
    """Main function"""
    args = parse_args()
    
    logger.info("Starting SadTalker inference...")
    logger.info(f"Audio: {args.audio_path}")
    logger.info(f"Image: {args.image_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    
    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        logger.error(f"Audio file not found: {args.audio_path}")
        return
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    result = run_inference(args.audio_path, args.image_path, args.output_dir, args.device, args)
    
    if result:
        logger.info("Inference completed successfully!")
        logger.info(f"Output video: {result['video_path']}")
        
        # Save result info
        info_path = os.path.join(args.output_dir, "inference_info.json")
        with open(info_path, "w") as f:
            json.dump({
                "audio_path": args.audio_path,
                "image_path": args.image_path,
                "video_path": result['video_path'],
                "timestamp": strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        logger.info(f"Inference info saved to: {info_path}")
    else:
        logger.error("Inference failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 