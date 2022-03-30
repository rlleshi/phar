
# phar

Make Porn Great Again

## Building datasets

### Build Video Dataset

1. Define the annotations @resources/annotations/annotations.txt
2. Create the clip dataset using `tools/data/generate_dataset.py`
3. Downgrade the quality of the videos using `tools/data/resize_videos.py`. Training will be much faster as resize overhead is removed.
4. Ggf. use `RepeatDataset` to further speed up training.

### Build Pose Dataset

1. Define the annotations @resources/annotations/annotations_pose.txt
2. Extract the pose information from the videos with `tools/analysis/pose_feasibility.py` or `tools/data/skeleton/generate_dataset_pose.py`
    - Best to extract with `pose_feasibility` as it will not save those poses with low confidence and it also gives feedback on the hardness of the dataset to extract poses.
3. Merge the poses into lists with `merge_pose` @`tools/misc.py`
4. Happy Training!

### Build Audio Dataset

1. Define the annotations @resources/annotations/annotations_audio.txt
2. Extract the audio from the videos with `mmaction2/tools/data/extract_audio.py`
    - `Stream map '0:a' matches no streams` means that the videos have no audio!
3. Extract spectogram features with `mmaction2/tools/data/build_audio_features.py`
4. Generate annotation list with `tools/data/audio/build_file_list.py`
5. Happy Training!
