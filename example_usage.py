#!/usr/bin/env python3
"""
Example usage of the SadTalker inference script
"""

import subprocess
import os
import sys

def run_example_1():
    """Example 1: Basic inference with audio file"""
    print("Example 1: Basic inference with audio file")
    
    cmd = [
        "python", "inference.py",
        "--audio_path", "./input_audio.wav",
        "--image_path", "./sadtalker_default.jpeg",
        "--output_dir", "./output/example1",
        "--device", "cuda",
        "--still_mode"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_example_2():
    """Example 2: Enhanced inference with base64 output"""
    print("Example 2: Enhanced inference with base64 output")
    
    cmd = [
        "python", "inference.py",
        "--audio_path", "./input_audio.wav",
        "--image_path", "./sadtalker_default.jpeg",
        "--output_dir", "./output/example2",
        "--device", "cuda",
        "--enhancer", "gfpgan",
        "--expression_scale", "1.2",
        "--preprocess", "full",
        "--save_base64"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_example_3():
    """Example 3: CPU mode with custom settings"""
    print("Example 3: CPU mode with custom settings")
    
    cmd = [
        "python", "inference.py",
        "--audio_path", "./input_audio.wav",
        "--image_path", "./sadtalker_default.jpeg",
        "--output_dir", "./output/example3",
        "--device", "cpu",
        "--batch_size", "5",
        "--preprocess", "crop"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def show_help():
    """Show help for the inference script"""
    print("Showing help for inference script:")
    subprocess.run(["python", "inference.py", "--help"])

if __name__ == "__main__":
    print("SadTalker Inference Script Examples")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "1":
            run_example_1()
        elif example == "2":
            run_example_2()
        elif example == "3":
            run_example_3()
        elif example == "help":
            show_help()
        else:
            print("Invalid example number. Use 1, 2, 3, or help")
    else:
        print("Usage: python example_usage.py [1|2|3|help]")
        print("")
        print("Examples:")
        print("  python example_usage.py 1    # Run example 1 (basic inference)")
        print("  python example_usage.py 2    # Run example 2 (enhanced with base64)")
        print("  python example_usage.py 3    # Run example 3 (CPU mode)")
        print("  python example_usage.py help # Show help")
        print("")
        print("Make sure you have:")
        print("  1. Installed dependencies: pip install -r requirements.txt")
        print("  2. Downloaded model checkpoints to ./checkpoints/")
        print("  3. Prepared an audio file (input_audio.wav)")
        print("  4. The default image file (sadtalker_default.jpeg)") 