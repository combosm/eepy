# eepy

## Installation

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

The decompressed file must end up at `models/shape_predictor_68_face_landmarks.dat` — that is
the path `scripts/camera.py` loads it from.

The other two model files (`opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt`)
are tracked in the repository and need no separate download.