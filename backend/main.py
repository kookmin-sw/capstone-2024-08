from fastapi import FastAPI, File, UploadFile, Form

from feedback import levenshtein
from fastapi.responses import JSONResponse, FileResponse
import tempfile
from feedback import stt
from create_script.user_script import create_user_script 
from create_script.user_script.schemas.gpt_sch import GptRequestSch, GptResponseSch
from fastapi.middleware.cors import CORSMiddleware
#from voice_converison import change_voice
from tts import infer
import text


app = FastAPI()
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
)


@app.post("/script", response_model= GptResponseSch)
async def create_script(req: GptRequestSch):
    script = await create_user_script.create_script_by_gpt(req)
    return {"script": script}


@app.post("/feedback/")
async def create_upload_file(sentence: str = Form(...), user_wav: UploadFile = File(...)):
    if not user_wav.filename.endswith('.wav'):
        return JSONResponse(status_code=400, content={"message": "Please upload WAV files only."})

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        # Write the uploaded file content to the temporary file
        content = await user_wav.read()
        tmp.write(content)
        tmp_path = tmp.name
        tmp.seek(0)
        user_trans = stt.transcribe_korean_audio(tmp_path)

    # TODO: Add guide audio later

    print(user_trans, sentence)

    cleaned_guide = text._clean_text(sentence, None)
    cleaned_user = text._clean_text(user_trans, None)
    similarity_percentage = levenshtein.dist(cleaned_guide, cleaned_user)
    # similarity_percentage = levenshtein.dist(guide, user)
    return {"similarity_percentage": similarity_percentage}


@app.post("/voice_guide/")
async def provide_voice_guide(sentence: str):
    # test

    # part-1: tts
    guide_audio = infer(sentence)


    # part-2: voice conversion
    change_voice("voice_converison/SPK064KBSCU001M001.wav", ["/home/ubuntu/forked/capstone-2024-08/backend/voice_converison/output_audio.wav"])
    print("conversion complete!!")
    return FileResponse("/home/ubuntu/forked/capstone-2024-08/backend/voice_converison/vc_out.wav", media_type="audio/wav")



@app.post("/test/")
async def test(sentence: str = Form(...), wavs: list[UploadFile] = File(...)):
    # Process received text and files here
    print(f"Received text: {text}")
    for wav_file in wavs:
        contents = await wav_file.read()
        print(f"Received file: {wav_file.filename}, size: {len(contents)} bytes")

    # Simulate processing and return a URL for a WAV file
    wav_url = "/home/ubuntu/capstone-2024-08/backend/voice_converison/SPK064KBSCU001M001.wav"
    return JSONResponse(status_code=200, content={"wav_url": wav_url})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
