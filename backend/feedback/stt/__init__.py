# from whisper_jax import FlaxWhisperPipline
# import jax.numpy as jnp
#
#
# def transcribe_korean_audio(file_path):
#     pipeline = FlaxWhisperPipline("openai/whisper-medium", dtype=jnp.bfloat16)
#     result = pipeline(file_path, task="transcribe")
#     return result["text"]

import whisper


def transcribe_korean_audio(file_path):
    model = whisper.load_model("medium", device="cuda")
    result = model.transcribe(file_path)
    print(result["text"])
    return result["text"]