#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify ultralytics library installation and basic usage
"""

def test_installation():
    """Test if ultralytics can be imported"""
    try:
        import ultralytics
        print(f"✓ Ultralytics version: {ultralytics.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ultralytics: {e}")
        return False

def test_basic_usage():
    """Test basic YOLO model usage"""
    try:
        from ultralytics import YOLO
        
        # Create a new YOLO model from scratch (small model)
        print("\nTesting basic YOLO model creation...")
        model = YOLO("yolo11n.yaml")  # Create a new model from YAML
        print("✓ Successfully created YOLO model from YAML")
        
        # Show model info
        print(f"✓ Model type: {type(model)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed basic usage test: {e}")
        return False

def test_pretrained_model():
    """Test loading a pretrained model (will download if not present)"""
    try:
        from ultralytics import YOLO
        
        print("\nTesting pretrained model loading...")
        # Load a lightweight pretrained model
        model = YOLO("./models/yolo26n-seg.pt")  # Load a pretrained model
        print("✓ Successfully loaded pretrained YOLO model")
        
        return True
    except Exception as e:
        print(f"✗ Failed pretrained model test: {e}")
        return False

def test_prediction():
    """Test making a prediction with a sample image"""
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        from PIL import Image
        
        print("\nTesting prediction on sample image...")
        
        # Create a simple test image using numpy
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Load model
        model = YOLO("./models/yolo26n-seg.pt")

        # Make prediction
        results = model(test_image)
        print("✓ Successfully made prediction on test image")
        
        # Show results info
        if results:
            result = results[0]
            print(f"✓ Prediction completed, result type: {type(result)}")
            
            # Check if boxes exist
            if result.boxes is not None:
                print(f"✓ Found {len(result.boxes)} detections")
            else:
                print("✓ No detections found (which is OK for random image)")
        
        return True
    except Exception as e:
        print(f"✗ Failed prediction test: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("Testing Ultralytics Library Installation and Basic Usage")
    print("="*60)
    
    # Test 1: Import
    success = test_installation()
    if not success:
        print("\n❌ Installation failed. Please install ultralytics using:")
        print("   pip install ultralytics")
        return
    
    # Test 2: Basic usage
    success = test_basic_usage()
    if not success:
        print("\n❌ Basic usage test failed.")
        return
    
    # Test 3: Pretrained model
    success = test_pretrained_model()
    if not success:
        print("\n❌ Pretrained model test failed.")
        return
    
    # Test 4: Prediction
    success = test_prediction()
    if not success:
        print("\n❌ Prediction test failed.")
        return
    
    print("\n" + "="*60)
    print("✅ All tests passed! Ultralytics is properly installed and working.")
    print("="*60)

if __name__ == "__main__":
    main()