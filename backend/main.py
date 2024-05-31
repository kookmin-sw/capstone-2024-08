from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import os
import shutil
from feedback import levenshtein
from fastapi.responses import JSONResponse, FileResponse
import tempfile
from feedback import stt
from create_script.user_script import create_user_script 
from create_script.user_script.schemas.gpt_sch import GptRequestSch, GptResponseSch
from fastapi.middleware.cors import CORSMiddleware
from voice_conversion import change_voice
from tts import infer
import text
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
from tts import utils
from tts.models import SynthesizerTrn
from text.symbols import symbols
from TTS.api import TTS



current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
tts_dir = os.path.join(current_dir, "tts")
config_path = os.path.join(tts_dir, "config", "nia22.json")
model_path = os.path.join(tts_dir, "vits_nia22.pth")
dummy_path = os.path.join(current_dir, "voice_conversion", "SPK014KBSCU004F002.wav")
pipeline = FlaxWhisperPipline("openai/whisper-medium", dtype=jnp.bfloat16)
_ = pipeline(dummy_path, task="transcribe")
hps = utils.get_hparams_from_file(config_path)
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cpu()
_ = net_g.eval()
_ = utils.load_checkpoint(model_path, net_g, None)
kkaguragzi = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")


app = FastAPI()


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
)

# Upload statics
app.mount("/static", StaticFiles(directory="/home/ubuntu/capstone-2024-08/backend"), name="static")


@app.post("/script", response_model= GptResponseSch)
async def create_script(req: GptRequestSch):
    script = await create_user_script.create_script_by_gpt(req)
    return {"script": script}


@app.post("/feedback/")
async def create_upload_file(sentence: str = Form(...), user_wav: UploadFile = File(...)):
    if not user_wav.filename.endswith('.wav'):
        return JSONResponse(status_code=400, content={"message": "Please upload WAV files only."})

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        content = await user_wav.read()
        tmp.write(content)
        tmp_path = tmp.name
        tmp.seek(0)
        user_trans = stt.transcribe_korean_audio(tmp_path, pipeline)

    cleaned_guide = text._clean_text(sentence, None)
    cleaned_user = text._clean_text(user_trans, None)
    similarity_percentage = levenshtein.dist(cleaned_guide, cleaned_user)
    return {"similarity_percentage": similarity_percentage, "pronunciation": user_trans}


@app.post("/voice_guide/")
async def provide_voice_guide(sentence: str = Form(...), wavs: list[UploadFile] = File(...)):
    temp_dir = tempfile.mkdtemp()
    user_voices_paths = []
    for wav in wavs:
        temp_file_path = os.path.join(temp_dir, wav.filename)
        with open(temp_file_path, 'wb+') as out_file:
            shutil.copyfileobj(wav.file, out_file)
        user_voices_paths.append(temp_file_path)
    guide_audio_path = infer(sentence, hps, net_g)
    output_voice_path = change_voice(kkaguragzi, guide_audio_path, user_voices_paths[0])
    shutil.rmtree(temp_dir)
    return JSONResponse(status_code=200, content={"wav_url": output_voice_path})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
