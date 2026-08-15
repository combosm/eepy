# Eepy - The AI Driving Assistant

Fatigue is a factor in a large share of road crashes, and unlike speeding or drunk driving
it's hard to self-detect. You don't usually notice you're too tired to drive safely until
it's too late. Eepy watches the driver and flags drowsiness as it happens,
instead of relying on the driver to catch it themselves.

## How it works

Each frame from the camera is run through a face detector (OpenCV) and a facial landmark
model (dlib) to locate 68 points on the driver's face. From those points, EEPY computes:

- **EAR (Eye Aspect Ratio)**: how open or closed the eyes are
- **MAR (Mouth Aspect Ratio)**: how open the mouth is, to catch yawning

Drowsiness is confirmed from *sustained* eye closure, optionally corroborated by recent
yawning. There's also a voice assistant you can query on demand from the same page.

## Installation

EEPY requires Python 3.10 or newer. LangChain 1.x no longer supports Python 3.9.

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

If you encountering an error with downloading `dlib`, you will need to download CMake and ensure it's added to your environmental path (there's a setting for this during installation).

Additionally, download the `shape_predictor_68_face_landmarks.dat` file. It is not tracked in
this repository (95 MB), so you need to fetch and decompress it yourself:

```sh
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 \
  https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 models/shape_predictor_68_face_landmarks.dat.bz2
```
