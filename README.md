# Seed-VC  

See [README.md](https://github.com/Plachtaa/seed-vc) for the original README.

## Setup

```console
wget "https://huggingface.co/thewh1teagle/seed-vc-heb/resolve/main/seed-vc-heb.7z"
7z x seed-vc-heb.7z
uv sync
uv pip install "torch==2.9.1" "torchaudio==2.9.1" "torchvision==2.9.1" --torch-backend=auto
```

## Run

```console
python inference.py --source <source-wav> --target <referene-wav> --output <output-dir> \
    --checkpoint <path-to-checkpoint> --config <path-to-config>
```