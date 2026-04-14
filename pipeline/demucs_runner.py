"""
Demucs wrapper — patches torchaudio.save() to use soundfile instead of torchcodec.

torchcodec requires FFmpeg shared libraries (DLLs) which are not always available
on Windows. This wrapper monkey-patches the save function before Demucs runs,
so stem WAV files are written via soundfile instead.
"""

import torchaudio
import soundfile as sf
import torch


def _soundfile_save(filepath, src, sample_rate, **kwargs):
    """Drop-in replacement for torchaudio.save using soundfile."""
    if isinstance(src, torch.Tensor):
        data = src.detach().cpu().numpy()
    else:
        data = src
    # torchaudio uses channels-first (C, T); soundfile expects (T, C)
    if data.ndim == 2:
        data = data.T
    sf.write(str(filepath), data, sample_rate, subtype="FLOAT")


torchaudio.save = _soundfile_save

# Now run Demucs with the patched save — CLI args are passed via sys.argv
import sys  # noqa: E402
from demucs.separate import main  # noqa: E402

main(sys.argv[1:])
