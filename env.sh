cd /export/fs04/a12/rhuang/forced_alignment

conda activate /export/fs04/a12/rhuang/anaconda/anaconda3/envs/espnet_gpu
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/:$PYTHONPATH

nvidia-smi
source /home/gqin2/scripts/acquire-gpu 1
echo $CUDA_VISIBLE_DEVICES

# https://github.com/NVIDIA/NeMo/tree/main/nemo_text_processing/inverse_text_normalization
## Install NeMo, which installs both nemo and nemo_text_processing package
BRANCH='r1.14.0'
python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[nlp]

# install Pynini for text normalization
wget https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo_text_processing/install_pynini.sh
bash install_pynini.sh

