# Self-Driving Car Simulation using YOLO and Computer Vision

This project implements a self-driving car simulation using state-of-the-art techniques in computer vision, deep learning, and autonomous driving. The system employs the YOLOv5 (You Only Look Once) object detection model for real-time detection of vehicles, pedestrians, and traffic signs. In addition, a Convolutional Neural Network (CNN) model is trained using behavioral cloning to predict steering angles based on video frames, enabling autonomous driving in a simulated environment.

---

## Tech Stack

This project utilizes the following technologies and libraries:

- **OpenCV**: For video processing, image manipulation, and lane detection.
- **YOLOv5**: For real-time object detection (vehicles, pedestrians, and traffic signs).
- **TensorFlow**: For training and deploying deep learning models.
- **Keras**: High-level neural networks API for building and training the behavioral cloning model.

---

## Project Overview

In this project, a self-driving car simulation system was developed using a combination of object detection, behavioral cloning, and autonomous navigation techniques. The main components of the project include:

1. **Real-Time Object Detection**: YOLOv5 is used to detect and classify various objects such as vehicles, pedestrians, and traffic signs in the camera feed of the simulated self-driving car.
2. **Steering Angle Prediction**: A CNN model is trained using behavioral cloning to predict the correct steering angle based on the video frames from the car's front-facing camera.
3. **Lane Detection and Curvature Estimation**: Using OpenCV, the system detects lanes on the road and calculates the curvature to assist in lane-keeping and path planning.
4. **Path Planning**: Based on the detected lanes and predicted steering angles, the car is able to follow a path autonomously.

---

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/KhushiNyati/self-driving-car-simulation.git
cd self-driving-car-simulation
pip install -r requirements.txt

### Overview
This project demonstrates the use of **YOLO (You Only Look Once)** for real-time object detection, along with **computer vision** techniques to create a self-driving car simulation. The system is designed to detect objects like pedestrians, vehicles, traffic signs, and lanes, enabling the car to navigate autonomously.

### Key Features
- **Real-time Object Detection** using YOLO to identify objects like cars, pedestrians, and traffic signs.
- **Lane Detection** to identify lanes and keep the vehicle centered.
- **Vehicle Control** to simulate steering and speed control based on detected objects and lanes.

### Technologies Used
- **YOLOv4** (for object detection)
- **OpenCV** (for image processing and video handling)
- **TensorFlow/Keras** (for machine learning models)
- **Python** (programming language)
- **NumPy** (for numerical operations)
- **Matplotlib** (for visualizing results)

## Usage
### 1 Run the simulation: After installing the dependencies, run the simulation by executing the following command:
python main.py
This will start the simulation environment where the self-driving car will navigate autonomously using YOLO for object detection and the trained CNN model for steering angle prediction.

### 2 Train the model: To retrain the behavioral cloning model, use the following command:

python train.py --data_path path_to_data --model_output path_to_save_model

## Key Features
Real-time Object Detection: YOLOv5 detects vehicles, pedestrians, and traffic signs, enabling the car to react to dynamic obstacles in real-time.

Lane Detection: The system identifies lanes on the road and uses this information to guide the car along the correct path.

Behavioral Cloning: The model predicts the car's steering angles based on video frames, enabling autonomous driving.

Curvature Estimation: Calculates the road curvature for better handling of sharp turns and path planning.

## Results and Performance
The trained model can navigate through a simulation environment, avoiding obstacles, following lanes, and obeying traffic signs.

Real-time object detection with YOLOv5 allows the car to detect and react to obstacles dynamically.

The steering angle predictions from the CNN model help the car follow the road safely and accurate
