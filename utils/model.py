import logging
import torch
import torchaudio

def load_model(device=None):
    logging.info(f"torch: {torch.__version__}")
    logging.info(f"torchaudio: {torchaudio.__version__}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logging.info(f"devide: {device}")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    sample_rate = bundle.sample_rate
    return model, labels, sample_rate, device
