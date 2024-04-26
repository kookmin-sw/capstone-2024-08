
import whisper


def transcribe_korean_audio(file_path):
    # 모델 로드 (한국어 포함 모델 선택)
    model = whisper.load_model("medium")
    result = model.transcribe(file_path)
    return result["text"]


# if __name__ == "__main__":
#     transcribe_korean_audio("backend/feedback/_samples/SPK064KBSCU001M001.wav")
