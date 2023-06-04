import numpy as np
import librosa
import torch
from torchvision import transforms as Tv
from torchaudio import transforms as Ta
import traceback

class SoundAugmentation:
    def __init__(
        self,
        sampling_rate=32000,
        n_fft=2048,
        ratio_harmonic=0.0,
        ratio_pitch_shift=0.0,
        ratio_percussive=0.0,
        ratio_time_stretch=0.0,
        range_harmonic_margin=[1, 3],
        range_n_step_pitch_shift=[-0.5, 0.5],
        range_percussive_margin=[1, 3],
        range_rate_time_stretch=[0.9, 1.1]
    ):
        # const
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.enable = {
            "pitch_shift": (ratio_pitch_shift > 0.0),
            "time_stretch": (ratio_time_stretch > 0.0),
            "harmonic": (ratio_harmonic > 0.0),
            "percussive": (ratio_percussive > 0.0)
        }
        self.config = {
            "harmonic": {
                "ratio": ratio_harmonic,
                "range_margin": range_harmonic_margin
            },
            "pitch_shift": {
                "ratio": ratio_pitch_shift,
                "range_n_step": range_n_step_pitch_shift
            },
            "percussive": {
                "ratio": ratio_percussive,
                "range_margin": range_percussive_margin
            },
            "time_stretch": {
                "ratio": ratio_time_stretch,
                "range_rate": range_rate_time_stretch
            },
        }

    def __call__(self, snd):
        return self.run(snd)

    def run(self, snd):
        # check length
        if len(snd) < self.n_fft:
            return snd

        # harmonic
        if self._is_applied(self.config["harmonic"]["ratio"]):
            harmonic_margin = np.random.uniform(
                *self.config["harmonic"]["range_margin"]
            )
            snd = librosa.effects.harmonic(snd, margin=harmonic_margin)
        # pitch shift
        if self._is_applied(self.config["pitch_shift"]["ratio"]):
            pitch_shift_n_step = np.random.uniform(
                *self.config["pitch_shift"]["range_n_step"]
            )
            snd = librosa.effects.pitch_shift(
                snd, n_steps=pitch_shift_n_step, sr=self.sampling_rate
            )
        # percussive
        if self._is_applied(self.config["percussive"]["ratio"]):
            percussive_mergin = np.random.uniform(
                *self.config["percussive"]["range_margin"]
            )
            snd = librosa.effects.percussive(snd, margin=percussive_mergin)
        # time stretch
        if self._is_applied(self.config["time_stretch"]["ratio"]):
            time_stretch_rate = np.random.uniform(
                *self.config["time_stretch"]["range_rate"]
            )
            snd = librosa.effects.time_stretch(snd, rate=time_stretch_rate)
        return snd

    def _is_applied(self, ratio):
        return np.random.choice([True,False], p=[ratio, 1-ratio])

class Fadein:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, x):
        fade_gain = np.random.uniform(
            1.0 / (x.shape[2] * self.ratio), 1.0
        )
        weights = torch.clip(
            fade_gain * torch.arange(1, x.shape[2] + 1),
            0.0, 1.0
        ).to(torch.float32)
        return x * weights

class Fadeout:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, x):
        fade_gain = np.random.uniform(
            1.0 / (x.shape[2] * self.ratio), 1.0
        )
        weights = torch.flip(
            torch.clip(
                fade_gain * torch.arange(1, x.shape[2] + 1),
                0.0, 1.0
            ),
            dims=[0]
        ).to(torch.float32)
        return x * weights

class Mixup:
    def __init__(self, alpha=0.01, device="cuda"):
        self.alpha = alpha
        self.rand_generator = torch.distributions.beta.Beta(alpha, alpha)

    def __call__(self, melspec, label):
        return self.run(melspec, label)

    def run(self, melspec, label):
        lam = self.rand_generator.sample()
        melspec_mixup = lam * melspec + (1 - lam) * melspec.roll(shifts=1, dims=0)
        label_mixup = lam * label + (1 - lam) * label.roll(shifts=1, dims=0)
        return melspec_mixup, label_mixup

class LabelSmoothing:
    def __init__(self, eps=0.01, n_class=265, device="cuda"):
        eyes = torch.eye(n_class)
        self.softlabels = torch.where(eyes<=0, eps/(n_class-1), 1-eps).to(torch.float32).to(device)

    def __call__(self, label):
        return self.run(label)

    def run(self, label):
        return torch.matmul(label.to(torch.float32), self.softlabels)

class SpecAugmentation:
    def __init__(self, config):
        self.config = config
        self.spec_transform = self.create_spec_transform()

    def create_spec_transform(self):
        augmentations = []
        if ("pitch_shift" in self.config.keys()) and (self.config["pitch_shift"] is not None):
            pitchshift = Tv.RandomApply([
                Tv.RandomAffine(
                    degrees=0,
                    translate=(0, self.config["pitch_shift"]["max"])
                )],
                p=self.config["pitch_shift"]["probability"]
            )
            augmentations.append(pitchshift)
        if ("time_shift" in self.config.keys()) and (self.config["time_shift"] is not None):
            timeshift = Tv.RandomApply([
                Tv.RandomAffine(
                    degrees=0,
                    translate=(self.config["time_shift"]["max"], 0)
                )],
                p=self.config["time_shift"]["probability"]
            )
            augmentations.append(timeshift)
        if ("freq_mask" in self.config.keys()) and (self.config["freq_mask"] is not None):
            freqmask = Tv.RandomApply(
                [Ta.FrequencyMasking(self.config["freq_mask"]["max"])],
                p=self.config["freq_mask"]["probability"]
            )
            augmentations.append(freqmask)
        if ("time_mask" in self.config.keys()) and (self.config["time_mask"] is not None):
            timemask = Tv.RandomApply(
                [Ta.TimeMasking(self.config["time_mask"]["max"])],
                p=self.config["time_mask"]["probability"]
            )
            augmentations.append(timemask)
        if ("fadein" in self.config.keys()) and (self.config["fadein"] is not None):
            fadein = Tv.RandomApply(
                [Fadein(self.config["fadein"]["max"])],
                p=self.config["fadein"]["probability"]
            )
            augmentations.append(fadein)
        if ("fadeout" in self.config.keys()) and (self.config["fadeout"] is not None):
            fadeout = Tv.RandomApply(
                [Fadeout(self.config["fadeout"]["max"])],
                p=self.config["fadeout"]["probability"]
            )
            augmentations.append(fadeout)
        if len(augmentations)==0:
            return None
        return Tv.Compose(augmentations)

    def __call__(self, melspec, label=None):
        return self.run(melspec, label)

    def run(self, melspec, label=None):
        if label is None:
            return self.spec_transform(melspec)
        melspec = self.spec_transform(melspec)
        return melspec, label
