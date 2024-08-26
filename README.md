# Features
1. Automatic text prompt generation via image captioning and classification
2. Prompt based object detection and segmentation
3. Automatic image and video labelling
4. Labeled image augmentation
5. Model training and fine-tuning

# Installation
1. Clone the repository: `git clone https://github.com/privvyledge/autolabeller.git`
2. Install dependencies: ` sudo apt update && sudo apt install -y python3-venv python3.10-venv ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6`
3. Create a virtual environment: `python3 -m venv autolabeller_env`
4. Activate the virtual environment:
    Windows: `autolabeller_env\Scripts\activate.bat`
    Linux/Mac: `source autolabeller_env/bin/activate`
5. Install cuda: 
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-4
   rm cuda-keyring_1.1-1_all.deb
   ```
6. Install python packages: `python3 -m pip install --no-cache-dir -r requirements.txt`. 
7. Install autodistill python packages: `python3 -m pip install  --use-deprecated=legacy-resolver -r autodistill_requirements.txt`.
8. Install SAM2
9. Install LabelStudio
10. Install LabelStudio ML Backend:
   ```bash
   git clone https://github.com/HumanSignal/label-studio-ml-backend.git
   cd label-studio-ml-backend
   python3 -m pip install -e .
   cd label_studio_ml/examples/segment_anything_2_image
   python3 -m pip install -r requirements.txt
   ```

## Todo
* Add image captioning and classification
* Add automatic prompt generation using image embeddings, captions or CLIP
