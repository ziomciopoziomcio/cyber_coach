# 🏋️‍♂️ Cyber Trainer – Intelligent Exercise Analysis System

**Cyber Trainer** is a computer-vision–based system designed to evaluate the correctness of physical exercises using two camera streams and a voice-controlled interface. Unlike smartwatches that only track repetitions or heart rate, Cyber Trainer focuses on **movement quality, posture alignment, and injury prevention**.

The system provides real-time feedback, detects incorrect technique, and allows the user to review their performance visually after completing each exercise set.

---

## 🎯 Project Overview

Cyber Trainer processes synchronized video streams from two cameras, extracts body keypoints, and analyzes motion patterns to detect common mistakes. It is suitable for both sports and rehabilitation environments.

### Key Features
- Real-time posture and movement analysis  
- Hands-free voice interaction  
- Post-exercise visual review  
- Detection of dangerous or inefficient movement patterns  
- High precision thanks to dual-camera setup

### Example Use Cases
- Weightlifting
- Rehabilitation exercises (e.g., post-injury physical therapy)
- Bodyweight training (e.g., push-ups, squats, lunges)
- Sports-specific movements (e.g., tennis serve, golf swing)
- Yoga and flexibility routines

---

## 🖥️ System Architecture

### Hardware Setup
- **Two smartphones** acting as IP cameras on tripods  
- **Wireless microphone** worn by the user  
- **Laptop** running the Cyber Trainer application

### Software Pipeline

1. **Camera Streaming**  
   - Two simultaneous video streams captured over the local network.

2. **Pose Estimation**  
   - MediaPipe / YOLOv11 models for keypoint extraction.

3. **Movement Analysis Module**  
   - Joint angle calculation  
   - Motion pattern recognition  
   - Detection of common postural errors (e.g., back rounding, knee valgus)  
   - Custom low-level computational algorithms  

4. **Voice Interaction**  
   - Speech recognition for commands  
   - Real-time spoken feedback and corrections  
   - Configurable instructions per exercise  

5. **Session Review UI**  
   - Keypoint overlays on recorded video  
   - Graphs of joint angles and ROM  
   - Highlighted error frames  

6. **Database Layer**  
   - Session logs, performance metrics, historical data  

---

## 🗣️ Voice Interface Examples

### Correct Workflow
- **User:** “Start deadlift analysis.”  
- **System:** “Ready when you are. Perform 5 repetitions.”  
- **System (during exercise):** “Keep your back straight.”  
- **System (after exercise):** “Set completed. View summary or continue?”

### Error Handling
- Unrecognized command  
- Poor audio quality / background noise  
- User moving out of camera range  
- Insufficient lighting  
- System suggests corrective steps in each scenario  

---

## 🛠️ Technologies Used

- **Python**  
- **OpenCV** – video capture and preprocessing  
- **MediaPipe / YOLOv11 / Roboflow** – pose detection  
- **SpeechRecognition** – voice command processing  
- **Pygame** – audio feedback and UI  
- **Database system (not chosen yet)** – session and metric storage  

---

## 📊 Testing & Evaluation

The system is tested across multiple aspects:

- **Accuracy** of pose estimation  
- **Performance** in real-time conditions  
- **Robustness** to lighting changes, occlusion, and camera angles  
- **Voice control reliability**  
- **Limitations** such as fast motion artifacts or tracking loss  

Evaluation includes timing tests, error metrics, visual comparisons, and user feedback.

---

## 📄 Documentation

The project includes detailed documentation with:

- System design and architecture  
- Mathematical background of algorithms  
- Implementation notes  
- Test results and performance metrics  
- Limitations and potential improvements  
- Final conclusions  
