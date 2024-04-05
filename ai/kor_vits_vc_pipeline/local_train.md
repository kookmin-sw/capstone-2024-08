# 로컬에서 학습하기

## 환경 설정

0. 필요한 것: `Python==3.8`, CMake 및 C/C++ 컴파일러, ffmpeg

1. 이 저장소를 복제합니다.
2. `pip install -r requirements.txt`를 실행하여 필요한 패키지를 설치합니다.
3. GPU 버전 PyTorch를 설치합니다. CUDA 11.6 또는 11.7이 설치되어 있어야 합니다.

```bash
# CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

4. monotonic align 빌드하기

```bash
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```

5. 파인튜닝 학습에 필요한 데이터를 다운로드

```bash
mkdir pretrained_models
# download data for fine-tuning
# vits pretrained 모델을 다운로드 할 것

# create necessary directories
mkdir video_data
mkdir raw_audio
mkdir denoised_audio
mkdir custom_character_voice
mkdir segmented_character_voice
```

G 모델을 `G_0.pth`, D 모델을 `D_0.pth`로 이름 바꾸고 `pretrained_models` directory 밑에 넣기
그 후 config.json 수정할 것

6. 사용자 음성 넣기

   #### Short audios

   1. readme 참고해서 .zip 파일로 만들기
   2. `./custom_character_voice/` 디렉토리 밑에 넣기
   3. `unzip ./custom_character_voice/custom_character_voice.zip -d ./custom_character_voice/` 실행하기

7 & 8. 사용자 음성으로 파인튜닝 코드 작성 및 추론 코드 추가 필요
