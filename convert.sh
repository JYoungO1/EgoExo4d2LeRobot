export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python egoexo4d2LeRobot.py \
    --json-dir /path/to/EgoExo4d_hand_annotations/ \
    --video-dir /path/to/EgoExo_video \
    --takes-path /path/to/takes.json \
    --output-dir /path/to/local \
    --aggregate-name egoexo_lerobot
