# EgoExo4d2LeRobot
This script converts hand pose annotations from the EgoExo4D EgoPose Benchmark into a format compatible with the LeRobot dataset.

# 1. LeRobot
The EgoExo conversion script is based on the LeRobot code provided by Hugging Face.  
You must first clone the official LeRobot GitHub repository:
EgoExo Convert code is based on the LeRobot code provided by Hugging Face, so you need to clone the LeRobot GitHub repository.
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```
Download the egoexo4d2LeRobot.py file and requirements.txt into the lerobot directory.

# 2. Setting
Create and activate a dedicated Conda environment, install dependencies, and set up LeRobot in editable mode:
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .
pip install -r requirements.txt
```

# 3. Run
Use tmux to run the conversion script in a background session, so the process continues even if your terminal disconnects:
```bash
 tmux new -s egoexo_convert './run_convert.sh'
```
