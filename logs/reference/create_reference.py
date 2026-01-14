import os

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import HubertModel

from rvc.lib.predictors.f0 import RMVPE


def cf0(f0):
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel = np.clip(
        (f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1,
        1,
        f0_bin - 1,
    )
    return np.rint(f0_mel).astype(int)


ref = os.path.join("logs", "reference", "reference.wav")
audio, sr = librosa.load(ref, sr=16000)
trimmed_len = (len(audio) // 320) * 320
audio = audio[:trimmed_len]
print("audio", audio.shape)

rmvpe_model = RMVPE(device="cpu", sample_rate=16000, hop_size=160)
f0 = rmvpe_model.get_f0(audio, filter_radius=0.03)
print("f0", f0.shape)

f0c = cf0(f0)
print("f0c", f0c.shape)

cv_path = os.path.abspath(os.path.join("rvc", "models", "embedders", "contentvec"))
spin2_path = os.path.abspath(os.path.join("rvc", "models", "embedders", "spin-v2"))

feats = torch.from_numpy(audio).to(torch.float32).to("cpu")
feats = torch.nn.functional.pad(feats.unsqueeze(0), (40, 40), mode="reflect")
feats = feats.view(1, -1)

with torch.no_grad():
    try:
        cv_model = HubertModel.from_pretrained(cv_path)
        cv_feats = cv_model(feats)["last_hidden_state"]
        cv_feats = cv_feats.squeeze(0).float().cpu().numpy()
        print("cv", cv_feats.shape)
        cv_output_dir = os.path.join("logs", "reference", "contentvec")
        os.makedirs(cv_output_dir, exist_ok=True)
        np.save(os.path.join(cv_output_dir, "feats.npy"), cv_feats)
    except Exception as e:
        print(f"Error with contentvec: {e}")

    try:
        spin2_model = HubertModel.from_pretrained(spin2_path)
        spin2_feats = spin2_model(feats)["last_hidden_state"]
        spin2_feats = spin2_feats.squeeze(0).float().cpu().numpy()
        print("spin-v2", spin2_feats.shape)
        spin2_output_dir = os.path.join("logs", "reference", "spin-v2")
        os.makedirs(spin2_output_dir, exist_ok=True)
        np.save(os.path.join(spin2_output_dir, "feats.npy"), spin2_feats)
    except Exception as e:
        print(f"Error with spin-v2: {e}")

np.save(os.path.join("logs", "reference", "pitch_coarse.npy"), f0c)
np.save(os.path.join("logs", "reference", "pitch_fine.npy"), f0)
