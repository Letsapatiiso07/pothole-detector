"""
Real-time Inference Module
Enhanced with better Grad-CAM and performance monitoring
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from utils import generate_grad_cam
from collections import deque

def run_inference(model_path, video_path, img_size, confidence_threshold=0.7):
    """
    Run real-time inference on video or webcam.
    
    Args:
        model_path: Path to trained model
        video_path: Path to video file or '0' for webcam
        img_size: Input image size for model
        confidence_threshold: Threshold for pothole detection
    """
    try:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Open video source
        if video_path == '0':
            cap = cv2.VideoCapture(0)
            print("Using webcam...")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"Using video file: {video_path}")
        
        if not cap.isOpened():
            print("Error: Cannot open video stream")
            return
        
        # FPS calculation using moving average
        fps_queue = deque(maxlen=30)
        detection_count = 0
        frame_count = 0
        
        print("\nStarting inference... Press 'q' to quit, 's' to save frame")
        print(f"Confidence threshold: {confidence_threshold}")
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1
            original_frame = frame.copy()
            
            # Preprocess frame
            img = cv2.resize(frame, (img_size, img_size))
            img_norm = img / 255.0
            img_exp = np.expand_dims(img_norm, axis=0)
            
            # Predict
            pred = model.predict(img_exp, verbose=0)[0][0]
            
            # Determine label and confidence
            is_pothole = pred > confidence_threshold
            label = 'POTHOLE DETECTED!' if is_pothole else 'Normal Road'
            confidence = pred if is_pothole else 1 - pred
            
            # Color coding
            color = (0, 0, 255) if is_pothole else (0, 255, 0)  # Red for pothole, green for normal
            
            if is_pothole:
                detection_count += 1
                
                # Generate Grad-CAM heatmap
                try:
                    heatmap = generate_grad_cam(model, img_norm)
                    
                    # Resize heatmap to match frame
                    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
                    
                    # Apply colormap
                    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), 
                                                       cv2.COLORMAP_JET)
                    
                    # Overlay heatmap on frame
                    frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
                    
                    # Draw bounding box around hottest region
                    thresh = (heatmap_resized > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(frame, "Anomaly Region", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Grad-CAM error: {e}")
                
                # Alert sound (optional - comment out if annoying)
                # print('\a')  # System beep
            
            # Display information
            # Main label
            cv2.putText(frame, label, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # FPS
            frame_time = time.time() - start_time
            fps = 1 / frame_time if frame_time > 0 else 0
            fps_queue.append(fps)
            avg_fps = np.mean(fps_queue)
            
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Detection count
            cv2.putText(frame, f"Detections: {detection_count}", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Frame count
            cv2.putText(frame, f"Frame: {frame_count}", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit | 's' to save", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Pothole Detector', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame to {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        print("\n=== Inference Summary ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Potholes detected: {detection_count}")
        print(f"Detection rate: {detection_count/frame_count*100:.2f}%")
        print(f"Average FPS: {np.mean(fps_queue):.2f}")
        
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback
        traceback.print_exc()
        raise