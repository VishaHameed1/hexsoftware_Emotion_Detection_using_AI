import cv2
import numpy as np
import math

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_emotion(face_roi, gray_face):
    """Detect multiple emotions based on facial features"""
    
    # Smile detection
    smiles = smile_cascade.detectMultiScale(face_roi, 1.7, 20)
    has_smile = len(smiles) > 0
    
    # Eye detection
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 10)
    has_eyes = len(eyes) >= 2
    
    # Face dimensions
    h, w = gray_face.shape
    
    # Mouth region (lower part of face)
    mouth_region = face_roi[int(h*0.6):int(h*0.9), :]
    
    # Calculate mouth aspect ratio (for surprise/anger)
    if len(mouth_region) > 0:
        mouth_black = np.sum(mouth_region < 50) / mouth_region.size
    else:
        mouth_black = 0
    
    # Eye brow region (for anger)
    brow_region = face_roi[int(h*0.2):int(h*0.35), :]
    brow_tension = np.sum(brow_region < 60) / brow_region.size if len(brow_region) > 0 else 0
    
    # Face symmetry and expression detection
    left_face = gray_face[:, :w//2]
    right_face = cv2.flip(gray_face[:, w//2:], 1)
    
    # Ensure same size for comparison
    min_width = min(left_face.shape[1], right_face.shape[1])
    left_face = left_face[:, :min_width]
    right_face = right_face[:, :min_width]
    
    symmetry_diff = np.mean(np.abs(left_face.astype(float) - right_face.astype(float)))
    
    # Emotion logic
    if has_smile and mouth_black < 0.3:
        if brow_tension > 0.4:
            emotion = "MISCHIEVOUS 😏"
            color = (255, 255, 0)
        else:
            emotion = "HAPPY 😊"
            color = (0, 255, 0)
    
    elif brow_tension > 0.5 and mouth_black > 0.4:
        emotion = "ANGRY 😠"
        color = (0, 0, 255)
    
    elif mouth_black > 0.5 and not has_smile:
        emotion = "SAD 😢"
        color = (255, 0, 0)
    
    elif mouth_black < 0.2 and has_eyes and symmetry_diff > 40:
        emotion = "SURPRISED 😲"
        color = (255, 255, 0)
    
    elif brow_tension > 0.3 and mouth_black > 0.3 and mouth_black < 0.5:
        emotion = "FEARFUL 😨"
        color = (255, 0, 255)
    
    elif not has_smile and mouth_black > 0.2 and mouth_black < 0.4:
        emotion = "NEUTRAL 😐"
        color = (128, 128, 128)
    
    else:
        emotion = "NEUTRAL 😐"
        color = (128, 128, 128)
    
    return emotion, color, has_smile

# Advanced emotion detection using face geometry
def analyze_facial_features(face_roi):
    """Analyze facial features for emotion detection"""
    h, w = face_roi.shape
    
    # Different regions of face
    regions = {
        'forehead': face_roi[0:int(h*0.3), :],
        'eyes': face_roi[int(h*0.3):int(h*0.5), :],
        'nose': face_roi[int(h*0.5):int(h*0.6), :],
        'mouth': face_roi[int(h*0.6):int(h*0.85), :],
        'chin': face_roi[int(h*0.85):h, :]
    }
    
    # Calculate intensity and texture for each region
    features = {}
    for name, region in regions.items():
        if region.size > 0:
            features[name] = {
                'mean': np.mean(region),
                'std': np.std(region),
                'dark_ratio': np.sum(region < 60) / region.size
            }
    
    return features

# Webcam start
cap = cv2.VideoCapture(0)

# Set resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("="*50)
print("🎭 ADVANCED EMOTION DETECTION SYSTEM")
print("="*50)
print("Detects: Happy | Sad | Angry | Surprised | Fearful | Neutral")
print("Press 'q' to quit")
print("Press 's' to save screenshot")
print("Press 'i' for detailed info")
print("-"*50)

# For tracking emotion history
emotion_history = []
frame_count = 0

while True:
    # Frame capture
    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection with different scales
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        # Face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect emotion
        emotion, color, is_smiling = detect_emotion(face_roi, face_roi)
        
        # Track emotion history
        emotion_history.append(emotion.split()[0])
        if len(emotion_history) > 50:
            emotion_history.pop(0)
        
        # Draw face rectangle with emotion color
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Background for text
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        
        # Display emotion
        cv2.putText(frame, emotion, (x+5, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Additional info
        if is_smiling:
            cv2.putText(frame, "😊", (x+w-30, y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display current emotion trend
    if len(emotion_history) > 10:
        from collections import Counter
        recent = Counter(emotion_history[-20:])
        if recent:
            common_emotion = recent.most_common(1)[0][0]
            cv2.putText(frame, f"Trend: {common_emotion}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press q=quit s=save i=info", (10, frame.shape[0]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show frame
    cv2.imshow('🎭 Multi-Emotion Detection', frame)
    
    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = cv2.getTickCount()
        filename = f"emotion_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    elif key == ord('i'):
        print("\n📊 Emotion Detection Info:")
        print(f"   Detected: {emotion}")
        print(f"   History: {emotion_history[-5:]}")
        if len(emotion_history) > 10:
            from collections import Counter
            stats = Counter(emotion_history)
            print("   Statistics:")
            for em, count in stats.most_common(3):
                print(f"     {em}: {count} times")
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print("\n✅ Session ended!")
if emotion_history:
    from collections import Counter
    print("\n📈 Final Emotion Statistics:")
    stats = Counter(emotion_history)
    total = len(emotion_history)
    for emotion, count in stats.most_common():
        percentage = (count/total)*100
        bar = "█" * int(percentage/2)
        print(f"   {emotion:12} : {bar:30} {percentage:5.1f}%")