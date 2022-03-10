
# phar

Make Porn Great Again

## Build the Dataset for Training

## Build Video Dataset

## Build Pose Dataset

1. Define the annotations @resources/annotations/annotations_pose.txt
2. Extract the pose information from the videos with `tools/analysis/pose_feasibility.py` or `tools/data/skeleton/generate_dataset_pose.py`
3. Merge the poses into lists with `merge_pose` @`tools/misc.py`
4. Happy Training!

## Build Audio Dataset

1. Define the annotations @resources/annotations/annotations_audio.txt
2. Extract the audio from the videos with `mmaction2/tools/data/extract_audio.py`
    - `Stream map '0:a' matches no streams` means that the videos have no audio!
3. Extract spectogram features with `mmaction2/tools/data/build_audio_features.py`
4. Generate annotation list with `tools/data/audio/build_file_list.py`
5. Happy Training!
