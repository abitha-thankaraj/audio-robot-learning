# Audio robot learning
That Sounds Right: Auditory Self-Supervision for Dynamic Robot Manipulation

[Project Page](https://audio-robot-learning.github.io/)

For robot rollouts and data collection: Refer [sound-supervised-bot](https://github.com/abitha-thankaraj/sound-supervised-bot)

## Installation Instructions
1. Clone the repo
```
git clone git@github.com:abitha-thankaraj/audio-robot-learning.git
cd audio-robot-learning/
```

2. Install the required python packages:
```
conda env create -f setup/environment.yaml
```
OR 
```
cd setup/
pip install -r requirements.txt
```

3. Install the `aurl` library
```
pip install -e . 
```

4. Download the dataset [here](https://drive.google.com/file/d/1bIdViF4ARJh1nkgbSrZMNBUqEjdQ-k5T/view?usp=sharing).
Update path to the data_dir in your dataset configs.

## Instructions
To pretrain the self supervised encoder for AuRL:

```
cd aurl
python pretrain.py dataset=<task_name>/mel ssl=<aurl_variant>

```

To finetune aurl:
```
cd aurl
python train.py encoder_dir=<ssl_encoder_save_dir> dataset=<task_name>/mel
```
