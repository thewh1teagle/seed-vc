"""
Batch voice conversion using Seed-VC.
Precomputes reference features once, prefetches audio while GPU is busy.

Usage:
    uv run examples/batch_rvc.py <input_folder> <output_folder> [--ref ref.wav] [--steps N]

Example:
    uv run examples/batch_rvc.py heb-female-audio-ipa2-v2/wav output-he/ --ref ref.wav
"""

import argparse
import queue
import threading
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from seed_vc_wrapper import SeedVCWrapper


def precompute_ref(vc: SeedVCWrapper, ref_path: str, f0_condition: bool):
    sr = 22050 if not f0_condition else 44100
    mel_fn = vc.to_mel if not f0_condition else vc.to_mel_f0
    inference_module = vc.model if not f0_condition else vc.model_f0

    ref_audio = librosa.load(ref_path, sr=sr)[0]
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(vc.device)
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

    mel2 = mel_fn(ref_audio.float())
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
    S_ori = vc._process_whisper_features(ref_waves_16k, is_source=False)

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = vc.campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = vc.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).to(vc.device)[None]
    else:
        F0_ori = None

    prompt_condition, _, _, _, _ = inference_module.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    return dict(mel2=mel2, style2=style2, prompt_condition=prompt_condition, F0_ori=F0_ori, sr=sr)


def preprocess_source(vc: SeedVCWrapper, source_path: str, f0_condition: bool, ref_cache: dict):
    """CPU-side preprocessing: load audio, resample, extract whisper features."""
    sr = ref_cache["sr"]
    mel_fn = vc.to_mel if not f0_condition else vc.to_mel_f0

    source_audio = librosa.load(source_path, sr=sr)[0]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(vc.device)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    S_alt = vc._process_whisper_features(converted_waves_16k, is_source=True)
    mel = mel_fn(source_audio.float())

    if f0_condition:
        F0_alt = vc.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)
        F0_alt = torch.from_numpy(F0_alt).to(vc.device)[None]
    else:
        F0_alt = None

    return dict(S_alt=S_alt, mel=mel, F0_alt=F0_alt, sr=sr)


def prefetch_worker(vc, files, f0_condition, ref_cache, q, prefetch=4):
    for source in files:
        try:
            data = preprocess_source(vc, str(source), f0_condition, ref_cache)
        except Exception as e:
            data = e
        q.put((source, data))
    q.put(None)  # sentinel


@torch.no_grad()
@torch.inference_mode()
def run_gpu(vc: SeedVCWrapper, preprocessed: dict, ref_cache: dict, f0_condition: bool,
            diffusion_steps=10, length_adjust=1.0, inference_cfg_rate=0.7):
    sr = ref_cache["sr"]
    mel2 = ref_cache["mel2"]
    style2 = ref_cache["style2"]
    prompt_condition = ref_cache["prompt_condition"]
    S_alt = preprocessed["S_alt"]
    mel = preprocessed["mel"]
    F0_alt = preprocessed["F0_alt"]

    bigvgan_fn = vc.bigvgan_model if not f0_condition else vc.bigvgan_44k_model
    inference_module = vc.model if not f0_condition else vc.model_f0
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_wave_len = vc.overlap_frame_len * hop_length

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)

    if f0_condition and F0_alt is not None:
        F0_ori = ref_cache["F0_ori"]
        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        median_log_f0_ori = torch.log(voiced_F0_ori + 1e-5).median()
        median_log_f0_alt = torch.log(voiced_F0_alt + 1e-5).median()
        shifted_log_f0_alt = log_f0_alt.clone()
        shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    else:
        shifted_f0_alt = None

    cond, _, _, _, _ = inference_module.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

        with torch.autocast(device_type=vc.device.type, dtype=torch.float16):
            vc_target = inference_module.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style2, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate
            )
            vc_target = vc_target[:, :, mel2.size(-1):]

        vc_wave = bigvgan_fn(vc_target.float())[0]
        processed_frames, previous_chunk, should_break, _, full_audio = vc._stream_wave_chunks(
            vc_wave, processed_frames, vc_target, overlap_wave_len,
            generated_wave_chunks, previous_chunk, is_last_chunk, False, sr
        )
        if should_break:
            return full_audio

    return np.concatenate(generated_wave_chunks)


def main():
    parser = argparse.ArgumentParser(description="Batch Seed-VC voice conversion")
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--ref", type=Path, default=Path("ref.wav"))
    parser.add_argument("--checkpoint", type=Path, default=Path("seed-vc-heb/model.pth"))
    parser.add_argument("--config", type=Path, default=Path("seed-vc-heb/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"))
    parser.add_argument("--f0", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    args.output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_folder.glob("*.wav"))
    if not files:
        print(f"No .wav files found in {args.input_folder}")
        return

    print("Loading model...")
    vc = SeedVCWrapper(checkpoint=str(args.checkpoint), config=str(args.config))

    print(f"Precomputing reference features from {args.ref}...")
    ref_cache = precompute_ref(vc, str(args.ref), args.f0)

    sr = ref_cache["sr"]
    print(f"Converting {len(files)} files (steps={args.steps})...")

    q = queue.Queue(maxsize=4)
    t = threading.Thread(target=prefetch_worker, args=(vc, files, args.f0, ref_cache, q), daemon=True)
    t.start()

    with tqdm(total=len(files), unit="file") as bar:
        while True:
            item = q.get()
            if item is None:
                break
            source, data = item
            if isinstance(data, Exception):
                tqdm.write(f"ERROR {source.name}: {data}")
                bar.update(1)
                continue
            try:
                result = run_gpu(vc, data, ref_cache, args.f0, diffusion_steps=args.steps)
                sf.write(str(args.output_folder / source.name), result, sr)
            except Exception as e:
                tqdm.write(f"ERROR {source.name}: {e}")
            bar.update(1)


if __name__ == "__main__":
    main()
