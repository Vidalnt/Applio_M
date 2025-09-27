from whisper.model import Whisper, ModelDimensions
from whisper.audio import log_mel_spectrogram
import torch
from torch import nn

class WhisperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.final_proj = None

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        self.final_proj = nn.Linear(dims.n_text_state, 768)
        self.model = Whisper(dims)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        del self.model.decoder
        # cut = len(self.model.encoder.blocks) // 4
        # cut = -1 * cut
        # del self.model.encoder.blocks[cut:]

    def forward(self, speech):
        ppgln = speech.shape[1] // 320
        mel = log_mel_spectrogram(speech[0]).to(speech.device)

        with torch.no_grad():
            ppg_raw = self.model.encoder(mel.unsqueeze(0))
            ppg_projected = self.final_proj(ppg_raw)
            ppg = ppg_projected.data.cpu().float().numpy()
            ppg = ppg[:, :ppgln, :]

        return {"last_hidden_state": ppg}