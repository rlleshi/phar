# phar
make porn great again


# Build the Dataset for Training
## Build Video Dataset

## Build Pose Dataset

## Build Audio Dataset

1. Define the annotations @resources/annotations/annotations_audio.txt
2. Extract the audio from the videos with `mmaction2/tools/data/extract_audio.py`
    - `Stream map '0:a' matches no streams` means that the videos have no audio!
3. Extract spectogram features with `mmaction2/tools/data/build_audio_features.py`
4. Generate annotation list with `tools/data/audio/build_file_list.py`
5. Happy Training!