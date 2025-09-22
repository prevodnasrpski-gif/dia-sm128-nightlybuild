Files you need to replace
model.py
Location:
<your project folder>\dia\model.py

This is the Dia project’s source file where we patched load_audio to use soundfile + DAC instead of TorchCodec.
METADATA
Location inside your virtual environment:
<your project folder>\.venv\Lib\site-packages\nari_tts-0.1.0.dist-info\METADATA

Here we edited the dependency list, replacing the hard-pinned versions with relaxed requirements:

Requires-Dist: soundfile>=0.13.1
Requires-Dist: torch>=2.8.0.dev
Requires-Dist: torchaudio>=2.8.0.dev
Requires-Dist: triton-windows>=3.3.0; sys_platform == "win32"
Requires-Dist: triton>=3.3.0; sys_platform == "linux"


✅ Replace those two files in those exact locations, and your patched setup will work with PyTorch 2.9/2.10 + CUDA 12.8 on your RTX 50xx.






🔧 Guide: Modifying nari-tts 0.1.0 Metadata and Environment
1. Patched nari_tts-0.1.0.dist-info

Replace

The original METADATA file inside:
.venv\Lib\site-packages\nari_tts-0.1.0.dist-info\METADATA

had hard-pinned dependencies, for example:

Requires-Dist: triton-windows==3.2.0.post18; sys_platform == "win32"
Requires-Dist: torch==2.6.*
Requires-Dist: torchaudio==2.6.*

This caused conflicts with:
PyTorch 2.9/2.10 nightly builds (needed for CUDA 12.8 support).
Your RTX 50xx GPU which only works with cu128 wheels.

TorchCodec/Triton mismatch.

✅ What we changed

We edited the METADATA to relax the version constraints.
Now it looks like this:

Requires-Dist: soundfile>=0.13.1
Requires-Dist: torch>=2.8.0.dev
Requires-Dist: torchaudio>=2.8.0.dev
Requires-Dist: triton-windows>=3.3.0; sys_platform == "win32"
Requires-Dist: triton>=3.3.0; sys_platform == "linux"


This allowed:
Newer PyTorch (2.9/2.10 nightly with cu128) to install.
Newer Triton (3.3.0+, 3.4.0) which supports CUDA 12.8 kernels and RTX 50xx cards.
SoundFile as the fallback audio loader (instead of TorchCodec).

2. Installed PyTorch nightly for CUDA 12.8

We used the official nightly wheels:
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

This ensured that Torch/Torchaudio are built with cu128 support, required for RTX 5070Ti.

3. Replaced TorchCodec with SoundFile
In dia/model.py:
Removed torchaudio.load (which internally called TorchCodec).

Added soundfile.read (sf.read) + manual resampling (torchaudio.functional.resample).
Forwarded tensors to DAC encoder.
This fully removed dependency on TorchCodec’s broken DLLs.

4. Installed FFmpeg 7 (Full Build)

Torch/Torchaudio/TorchCodec require FFmpeg 4–7.
You had FFmpeg 8 (via winget), which is unsupported.

Steps we did:
Downloaded the full build of FFmpeg 7 (win64, GPL).
Example: ffmpeg-n7.0-7-gd38bf5e08e-win64-gpl-7.0.zip

Extracted it into C:\ffmpeg\bin.
Updated Windows PATH:
Added C:\ffmpeg\bin.
Removed / moved down C:\ProgramData\miniconda3\Library\bin\ffmpeg.exe.

Verified with:
where ffmpeg
✅ Only C:\ffmpeg\bin\ffmpeg.exe should be listed.

📌 Summary
Metadata (METADATA in nari_tts-0.1.0.dist-info) patched → no more hard-pinned dependencies.
Torch/Torchaudio/Triton upgraded to work with CUDA 12.8 (cu128).
TorchCodec replaced by SoundFile loader + DAC encode pipeline.
FFmpeg 7 full build installed and properly set in PATH (only one version active).

With this setup:
Dia now runs inference on your GPU (RTX 50xx) with PyTorch 2.10 nightly + cu128.
TorchCodec issues are bypassed.

No more conflicts between nari-tts and Triton.












🔧 Modification Guide for Dia (Nari-TTS Integration Fix)
1. Removed TorchCodec dependency
Originally, torchaudio.load internally relied on TorchCodec to decode audio.
TorchCodec was failing (libtorchcodec_core*.dll not loading) because of incompatibilities with PyTorch 2.9/2.10 + CUDA 12.8.
We removed TorchCodec usage and replaced it with SoundFile (sf.read) plus manual resampling.

📍 File: dia/model.py
📍 Function: load_audio

2. Rewritten load_audio
The load_audio function now:
Uses soundfile.read → loads waveform as (T, C) (time × channels).
Converts to PyTorch tensor (C, T).
Resamples to DEFAULT_SAMPLE_RATE (44.1 kHz) using torchaudio.functional.resample.
Mixes down to mono if multiple channels.
Passes the tensor to DAC encoder (self._encode).
✅ This ensures that instead of raw waveform, the model still receives the encoded [T, C] features it expects.

4. Helper methods for DAC
We confirmed and kept:
_encode → wraps self.dac_model.encode, converting audio tensors into discrete codebook representations.
_decode → wraps self.dac_model.decode, turning codebook tokens back into waveform.

📍 File: dia/model.py
📍 Functions: _encode, _decode

These methods now form the core of the replacement for TorchCodec’s old decoding/encoding pipeline.
4. Changes in generate workflow
generate still calls self.load_audio(audio_prompt) if the input is a path.
Because load_audio now calls _encode, the output is always the discrete DAC-coded sequence that _prepare_audio_prompt expects.
This fixed earlier shape errors (tuple has no attribute 'shape' and mismatched [samples] vs [tokens, dim]).

5. Fallback behavior
If no DAC model is loaded (self.dac_model is None), load_audio raises a RuntimeError.
This prevents the model from running with incomplete configuration.
🗂️ Where Things Are Now
dia/model.py
load_audio: replaced TorchCodec with SoundFile + DAC encode.
_encode, _decode: direct interface to DAC model.
generate: unchanged externally, but now works correctly with the new audio pipeline.
External dependencies
soundfile (libsndfile) → handles waveform loading.
torchaudio.functional.resample → still used for resampling.
dac → still required, provides audio codec (replaces TorchCodec).

✅ Summary
Before: torchaudio.load → TorchCodec → waveform → model
Now: soundfile.read → tensor → resample → mono → DAC encode → model
This keeps the model happy, avoids TorchCodec issues, and maintains compatibility with PyTorch 2.9/2.10 + CUDA 12.8.


