#!/usr/bin/env python3
"""
Minimal voice conversion example.
Converts source audio to match the voice of target audio.
"""

import os
import sys
import argparse
import warnings
import torch
import torchaudio
import librosa
import numpy as np
import yaml

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
warnings.simplefilter('ignore')

from modules.commons import build_model, load_checkpoint, recursive_munch
from hf_utils import load_custom_model_from_hf
from modules.campplus.DTDNN import CAMPPlus
from modules.audio import mel_spectrogram

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    fp16 = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    fp16 = False  # MPS doesn't support fp16 autocast
else:
    device = torch.device("cpu")
    fp16 = False


def crossfade(chunk1, chunk2, overlap):
    """Apply crossfade between two audio chunks."""
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def load_models():
    """Load all required models for voice conversion."""
    print("Loading models...")
    
    # Load DiT model
    dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    )
    
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model, None, dit_checkpoint_path,
        load_only_params=True, ignore_modules=[], is_distributed=False
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load CAMPPlus for speaker embedding
    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    # Load BigVGAN vocoder
    from modules.bigvgan import bigvgan
    bigvgan_name = model_params.vocoder.name
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)

    # Load Whisper for semantic features
    from transformers import AutoFeatureExtractor, WhisperModel
    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def semantic_fn(waves_16k):
        ori_inputs = whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True
        )
        ori_input_features = whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
        ).to(device)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
        return S_ori

    # Mel spectrogram function
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    print("Models loaded successfully!")
    return model, semantic_fn, bigvgan_model, campplus_model, to_mel, sr, hop_length


@torch.no_grad()
def convert_voice(source_path, target_path, output_path="audio.wav", 
                   diffusion_steps=25, length_adjust=1.0, inference_cfg_rate=0.7):
    """
    Convert source audio to match target voice.
    
    Args:
        source_path: Path to source audio file (to convert)
        target_path: Path to target audio file (reference voice to clone)
        output_path: Path to save output audio
        diffusion_steps: Number of diffusion steps (default: 25)
        length_adjust: Length adjustment factor (default: 1.0)
        inference_cfg_rate: Inference CFG rate (default: 0.7)
    """
    # Load models
    model, semantic_fn, vocoder_fn, campplus_model, mel_fn, sr, hop_length = load_models()
    
    # Load audio files
    print(f"Loading source audio: {source_path}")
    source_audio = librosa.load(source_path, sr=sr)[0]
    print(f"Loading target audio: {target_path}")
    ref_audio = librosa.load(target_path, sr=sr)[0]

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)  # Limit to 25s

    # Resample to 16kHz for feature extraction
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    
    # Extract semantic features (handle long audio by chunking)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        # Process long audio in chunks
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat([
                    buffer, 
                    converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                ], dim=-1)
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    # Compute mel spectrograms
    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    # Extract speaker style features
    feat2 = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # Length regulation
    cond, _, _, _, _ = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=None
    )
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
    )

    # Process in chunks for long audio
    max_context_window = sr // hop_length * 30
    max_source_window = max_context_window - mel2.size(2)
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length
    
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    print("Converting voice...")
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        
        # Use autocast only for CUDA, not for MPS or CPU
        if device.type == "cuda":
            # Voice Conversion
            with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
                vc_target = model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2, style2, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate
                )
        else:
            # For MPS/CPU, run without autocast
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style2, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate
            )
        vc_target = vc_target[:, :, mel2.size(-1):]
        
        vc_wave = vocoder_fn(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    
    # Concatenate all chunks
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
    
    # Save output
    print(f"Saving output to: {output_path}")
    torchaudio.save(output_path, vc_wave.cpu(), sr)
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Minimal voice conversion example - Convert source audio to match target voice"
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to source audio file (to convert)"
    )
    parser.add_argument(
        "target",
        type=str,
        help="Path to target audio file (reference voice to clone)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="audio.wav",
        help="Output audio file path (default: audio.wav)"
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=25,
        help="Number of diffusion steps (default: 25)"
    )
    parser.add_argument(
        "--length-adjust",
        type=float,
        default=1.0,
        help="Length adjustment factor (default: 1.0)"
    )
    parser.add_argument(
        "--inference-cfg-rate",
        type=float,
        default=0.7,
        help="Inference CFG rate (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    convert_voice(
        args.source,
        args.target,
        args.output,
        args.diffusion_steps,
        args.length_adjust,
        args.inference_cfg_rate
    )


if __name__ == "__main__":
    main()
