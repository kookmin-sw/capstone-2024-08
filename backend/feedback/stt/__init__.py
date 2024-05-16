def transcribe_korean_audio(file_path, pipeline):
    result = pipeline(file_path, task="transcribe")
    return result["text"]

# import whisper
#
#
# def transcribe_korean_audio(file_path):
#     model = whisper.load_model("medium", device="cuda")
#     result = model.transcribe(file_path)
#     print(result["text"])
#     return result["text"]