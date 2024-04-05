# VC 파이프라인 설명

이 저장소의 파이프라인은 다양한 음성 업로드 옵션을 지원합니다. 사용 가능한 데이터에 따라 하나 이상의 옵션을 선택할 수 있습니다.

### 짧은 오디오 파일

단일 .zip 파일로 압축된 짧은 오디오 파일은 다음과 같은 파일 구조여야 합니다:

```
Your-zip-file.zip
├───Character_name_1
│   ├───xxx.wav
│   └───zzz.wav
├───Character_name_2
│   ├───xxx.wav
│   └───zzz.wav
├───...
└───Character_name_n
    ├───xxx.wav
    └───zzz.wav
```

오디오 파일의 형식은 중요하지 않으나 다음의 품질 요구 사항을 충족해야 합니다:

- 길이: 2초 이상, 10초 이하
- 배경 소음 최소화

각 캐릭터에 대해 최소 10개의 파일이 있어야 하며, 20개 이상을 권장합니다.

### 긴 오디오 및 비디오 파일

- 캐릭터 이름으로 된 긴 오디오 및 비디오 파일은 하나의 캐릭터 음성만 포함해야 합니다. 배경 소음이 있어도 괜찮습니다. 자동으로 제거해주거든요 ㅎㅅㅎ
- 파일 이름 형식: `{CharacterName}_{random_number}.wav`
  (예: Diana_234135.wav, MinatoAqua_234252.wav)

---

# 추가할 것

- vc pipeline을 위해 `preprocess.py` 수정 필요
- 아래 패키지는 실행한 결과 `requirements.txt` 에서 설치가 안됐음. 만약 같은 문제 발생 시 수동으로 설치

```shell
jamo
phonemizer
jamotools
unidecode
g2pk
kiwipiepy
```

- ai hub에서 제공하는 모델은 https://drive.google.com/file/d/1hmTiFMVaB6i6adbJqVx92k5xQwBhj1Km/view 에서 다운 가능
