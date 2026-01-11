# Seed-VC  

See [README.md](https://github.com/Plachtaa/seed-vc) for the original README.

## Setup

```console
wget "https://huggingface.co/thewh1teagle/seed-vc-heb/resolve/main/seed-vc-heb.7z"
7z x seed-vc-heb.7z
uv sync
uv pip install "torch==2.9.1" "torchaudio==2.9.1" "torchvision==2.9.1" --torch-backend=auto
```

## Train

Dataset can contain ~10 minutes of audio. training took ~5 minutes on DGX Spark.

```console
uv run python train.py \
  --config ./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
  --dataset-dir ./dataset \
  --run-name hebrew-voice \
  --batch-size 2 \
  --max-steps 500 \
  --max-epochs 1000 \
  --save-every 250 \
  --num-workers 0
```

## Inference

```console
uv run python inference.py \
  --source ./source.wav \
  --target ./reference.wav \
  --output ./output \
  --checkpoint ./runs/hebrew-voice/ft_model.pth \
  --config ./runs/hebrew-voice/config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
  --diffusion-steps 25 \
  --fp16 True
```