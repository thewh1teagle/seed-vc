"""
wget https://huggingface.co/thewh1teagle/seed-vc-heb/resolve/main/seed-vc-heb.7z
7z x seed-vc-heb.7z

wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/male1.wav

Note: if uv fails to build scipy from source, install gfortran:
    sudo apt install gfortran
"""

import soundfile as sf
from seed_vc_wrapper import SeedVCWrapper


def apply_rvc(source, reference, output="output.wav", f0_condition=False, checkpoint=None, config=None):
    """
    Apply voice conversion using Seed-VC.
    
    Args:
        source: Path to source audio file
        reference: Path to reference audio file
        output: Path to output audio file (default: "output.wav")
        f0_condition: Whether to use F0 conditioning (default: False)
        checkpoint: Path to custom model checkpoint (optional)
        config: Path to custom config file (optional)
    
    Returns:
        Path to the output audio file
    """
    # Initialize the model wrapper
    vc = SeedVCWrapper(checkpoint=checkpoint, config=config)
    
    # Perform voice conversion (non-streaming mode)
    result_generator = vc.convert_voice(
        source=source,
        target=reference,
        f0_condition=f0_condition,
        stream_output=False
    )
    
    # Get the final result from the generator
    try:
        next(result_generator)
    except StopIteration as e:
        result = e.value
    
    # Save the result
    sr = 44100 if f0_condition else 22050
    sf.write(output, result, sr)
    
    return output


if __name__ == "__main__":
    # Basic usage example with custom model
    output_path = apply_rvc(
        source="target.wav",
        reference="ref.wav",
        checkpoint="seed-vc-heb/model.pth",
        config="seed-vc-heb/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    )
    print(f"Voice conversion complete! Output saved to: {output_path}")