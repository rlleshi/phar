
# phar

Make Porn Great Again

## Building datasets

### Build Video Dataset

1. Define the annotations @resources/annotations/annotations.txt
2. Create the clip dataset using `src/data/generate_dataset.py`
3. Downgrade the quality of the videos using `src/data/resize_videos.py`. Training will be much faster as resize overhead is removed.
4. Ggf. use `RepeatDataset` to further speed up training.
5. Use `mmaction2/src/analysis/check_videos.py` to check if the dataset is valid.

### Build Pose Dataset

1. Define the annotations @resources/annotations/annotations_pose.txt
2. Extract the pose information from the videos with `src/analysis/pose_feasibility.py` or `src/data/skeleton/generate_dataset_pose.py`
    - Best to extract with `pose_feasibility` as it will not save those poses with low confidence and it also gives feedback on the hardness of the dataset to extract poses.
3. Merge the poses into lists with `merge_pose` @`src/misc.py`

### Build Audio Dataset

1. Define the annotations @resources/annotations/annotations_audio.txt
2. Extract the audio from the videos with `mmaction2/src/data/extract_audio.py`
    - `Stream map '0:a' matches no streams` means that the videos have no audio!
3. Ggf. filter audios based on their loudness with `src/analysis/audio_filter.py`
4. Extract spectogram features with `mmaction2/src/data/build_audio_features.py`
5. Generate annotation list with `src/data/audio/build_file_list.py`
