import torch, torchaudio
from typing import List
import os


def change_voice(knn_vc, src: str, ref: List[str]):
    """
    src: path to 16kHz, single-channel, source waveform
    ref: list of paths to all reference waveforms (each must be 16kHz, single-channel) from the target speaker
    """
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    vc_out_path = os.path.join(current_dir, "vc_out.wav")
    query_seq = knn_vc.get_features(src)
    matching_set = knn_vc.get_matching_set(ref)
    out_wav = knn_vc.match(query_seq, matching_set, topk=4)
    torchaudio.save(vc_out_path, out_wav[None], 16000)
    return "http://ec2-13-124-219-249.ap-northeast-2.compute.amazonaws.com/static/vc_out.wav"
