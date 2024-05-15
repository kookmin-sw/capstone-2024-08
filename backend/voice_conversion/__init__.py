import torch, torchaudio
from typing import List


def change_voice(src: str, ref: List[str]):
    """
    src: path to 16kHz, single-channel, source waveform
    ref: list of paths to all reference waveforms (each must be 16kHz, single-channel) from the target speaker
    """
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')
    query_seq = knn_vc.get_features(src)
    matching_set = knn_vc.get_matching_set(ref)
    out_wav = knn_vc.match(query_seq, matching_set, topk=4)
    torchaudio.save("/home/ubuntu/forked/capstone-2024-08/backend/voice_converison/vc_out.wav", out_wav[None], 16000)
    return "http://ec2-13-124-219-249.ap-northeast-2.compute.amazonaws.com/static/vc_out.wav"
