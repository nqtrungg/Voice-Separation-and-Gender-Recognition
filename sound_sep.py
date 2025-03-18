from pyannote.database.util import load_rttm
from pyannote.audio.core.io import Audio
from pyannote.audio import Pipeline
import torch
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio.pipelines.utils.hook import ProgressHook

class SoundSeparation:
    default_sample_rate = 16000

    def __init__(self, config_path="config.yaml"):
        self.audio = Audio(sample_rate=self.default_sample_rate)
        self.pipeline = Pipeline.from_pretrained(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(device=self.device)
        self.pipeline.instantiate({
            "segmentation": {
                "min_duration_off": 0.0,
                "threshold": 0.5,
            },
            "clustering": {
                "method": "centroid",
                "threshold": 0.68,
                "min_cluster_size": 60,
            },
            "separation": {
                "leakage_removal": True,
                "asr_collar": 0.0,
            }
        })
        self.metric = DiarizationErrorRate()

    def read_audio_file(self, file_path):
        mixture, sample_rate = self.audio(file=file_path)
        return mixture, sample_rate

    def get_sample_rate(self, audio_file):
        _, sample_rate = self.audio(file=audio_file)
        return sample_rate

    def separate_sound(self, audio_file, sample_rate):
        with ProgressHook() as hook:
            diarization, sources_hat = self.pipeline({"waveform": audio_file, "sample_rate": sample_rate}, hook=hook)
        return diarization, sources_hat

# Example usage:
# sound_sep = SoundSeparation(config_path="config.yaml")
# mixture, sample_rate = sound_sep.read_audio_file("untitled.wav")
# diarization, sources_hat = sound_sep.separate_sound(mixture, sample_rate)