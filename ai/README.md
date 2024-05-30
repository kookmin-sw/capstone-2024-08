## Environment
```bash
conda create -n loro python=3.10
conda activate loro

cd vits2_pytorch
sudo apt-get install espeak
sudo pip3 install --ignore-installed -r requirements.txt
```
<br/>

## vscode에서 Jupyter notebook 사용하는 경우
```bash
pip3 install jupyter
pip3 install ipykernel
python3 -m ipykernel install --user --name loro --display-name loro
```
<br/>

## Dataset
뉴스 대본 및 앵커 음성 데이터셋 다운로드
- [AI Hub 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71557)
```
└── ai
    └── dataset
        ├── VL
        │   └── SPK003
        │       └── SPK003YTNSO162
        │           └── SPK003YTNSO162F001.json
        └── VS
            └── SPK003
                └── SPK003YTNSO162
                    └── SPK003YTNSO162F001.wav
```
<br/>

train, val, test 파일 리스트 생성


```bash
cd vits2_pytorch
ln -s /path/to/dataset/VS DUMMY3

cd sfen
ln -s /path/to/dataset/VS DUMMY3

python3 preprocess_filelist.py 
```

<br/>

# SFEN
<br/>

## Train

```bash
cd ai/sfen
python3 train.py --config ckpt/config.json
```
<br/>

## Fine-tuning

```bash
python3 train.py --fine_tuning True --config ckpt/config.json
```
<br/>

## Inference

```bash
python3 inference.py --checkpoint_file [vae checkpoint file path]
```
<br/>

## Clustering

```bash
python3 clustering.py --config path/to/ckpt/config.json
```

<br/><br/><br/>


# VITS2
<br/>

## Preprocess
```bash

cd monotonic_align
python setup.py build_ext --inplace

cd ..
python3 preprocess.py --text_index 2 --filelists filelists/loro_audio_sid_text_train_filelist.txt filelists/loro_audio_sid_text_val_filelist.txt filelists/loro_audio_sid_text_test_filelist.txt

```
<br/>

## Train
```bash
python3 train_ms.py -c configs/vits2_loro_base.json -m loro 
```
<br/>

## Inference
```bash
```