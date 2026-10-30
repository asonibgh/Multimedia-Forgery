import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS (must be set before CUDA init)
import glob
import json
import time
import math
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from tqdm import tqdm
import cv2
import librosa
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score,
    det_curve
)

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    SEED = 42
    DATASET_PATH = "/content/drive/MyDrive/DFDC-Shortcut/DFDC Dataset"
    OUTPUT_DIR = "/content/drive/MyDrive/IMPROVED_ENSEMBLE_OUTPUT"

    NUM_FRAMES = 20
    FRAME_SIZE = 224
    AUDIO_SR = 22050
    AUDIO_DURATION = 3.0

    BATCH_SIZE = 24
    EPOCHS = 200
    BASE_LR = 3e-5
    WEIGHT_DECAY = 5e-4
    GRADIENT_CLIP = 0.5
    EARLY_STOPPING_PATIENCE = 15
    MIN_DELTA = 0.001
    LABEL_SMOOTHING = 0.1

    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.5
    USE_TTA = True
    TTA_ITERATIONS = 3
    USE_MIXED_PRECISION = True
    CACHE_DATA = True
    MAX_CACHE_SIZE = 3000

    ENSEMBLE_WEIGHTS = [0.45, 0.20, 0.35]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = USE_MIXED_PRECISION and DEVICE == "cuda"
    NUM_WORKERS = 0

    # ---- figure / reporting options ----
    DPI = 600                 # saved resolution
    SHOW_PLOTS = True         # display every figure before it is saved
    SAVE_PDF = True           # also save vector PDF (best for LaTeX papers)
    SHOW_TITLES = True        # set False if your captions carry the title
    BOOTSTRAP_ITERS = 1000    # for 95% confidence intervals
    BENCH_BATCH_SIZES = (1, 2, 4, 8, 16, 24)

    # ---- reproducibility ----
    SEEDS = [1, 42, 97, 321]          # training seeds (weights init, augmentation, data order)
    PRIMARY_SEED = 42                 # seed used for the single-run deep-dive figures + efficiency benchmark
    SPLIT_SEED = 42                   # FIXED train/val/test split for every seed (=> paired statistics are valid)
    SKIP_FINISHED_TRAINING = True     # a seed with train_done.json is not re-trained (Colab-safe)
    AUTO_RESUME = True                # resume an interrupted seed from latest_checkpoint.pth (no prompt)
    REPEAT_EVAL_RUNS = 2              # repeat test inference N times per seed (repeatability check)
    REPEAT_TRAIN_CHECK = False        # True: re-train the primary seed a 2nd time from scratch (costly!)

    # ---- four thresholds (chosen on validation, evaluated on test) ----
    THRESHOLD_STRATEGIES = ['default_0.5', 'youden_val', 'max_f1_val', 'eer_val']

    # ---- ensemble weights [video, audio, fusion] compared post-hoc ----
    WEIGHT_CONFIGS = {
        'Default': [0.45, 0.20, 0.35], 'Equal': [1, 1, 1],
        'Video-only': [1, 0, 0], 'Audio-only': [0, 1, 0], 'Fusion-only': [0, 0, 1],
        'Video+Fusion': [0.5, 0, 0.5], 'Video-heavy': [0.6, 0.1, 0.3], 'Audio-heavy': [0.2, 0.5, 0.3],
    }

    # ---- robustness ----
    RUN_ROBUSTNESS = True
    ROBUSTNESS_SEEDS = [42]           # use SEEDS for all seeds (4x more passes over the test set)
    ROBUSTNESS_USE_TTA = False        # clean + perturbed passes without TTA (faster, isolates the perturbation)
    REUSE_CACHED_ROBUSTNESS = True

def set_seed(seed=Config.SEED):
    """Seed every RNG and force deterministic kernels (needed for repeatability)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

set_seed()

def amp_ctx():
    return autocast(enabled=Config.USE_AMP)

def audio_time_frames():
    """Number of mel frames produced by librosa (center=False, n_fft=1024, hop=512)."""
    return 1 + (int(Config.AUDIO_SR * Config.AUDIO_DURATION) - 1024) // 512

def cuda_sync():
    if Config.DEVICE == "cuda":
        torch.cuda.synchronize()

# =====================================================================
# DATA AUGMENTATION
# =====================================================================

class VideoAugmentation:
    @staticmethod
    def apply_spatial_aug(frames, prob=0.5):
        if random.random() > prob:
            return frames
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[3])
        if random.random() < 0.5:
            frames = torch.clamp(frames * random.uniform(0.8, 1.2), -2.5, 2.5)
        if random.random() < 0.3:
            frames = frames + (torch.randn(1, 3, 1, 1) * 0.1).to(frames.device)
        return frames

    @staticmethod
    def apply_temporal_aug(frames, prob=0.3):
        if random.random() > prob:
            return frames
        if random.random() < 0.5:
            mask = torch.rand(frames.size(0)) > 0.1
            if mask.sum() > frames.size(0) // 2:
                frames[~mask] = frames[mask][0]
        if random.random() < 0.3:
            perm = torch.randperm(frames.size(0))
            max_shift = min(3, frames.size(0) // 4)
            shifted = torch.clamp(perm + torch.randint(-max_shift, max_shift, perm.shape), 0, frames.size(0) - 1)
            frames = frames[shifted]
        return frames

class AudioAugmentation:
    @staticmethod
    def apply_audio_aug(audio, prob=0.5):
        if random.random() > prob:
            return audio
        if random.random() < 0.5:
            audio = audio + torch.randn_like(audio) * 0.01
        if random.random() < 0.3 and audio.size(2) > 10:
            sz = random.randint(1, max(1, audio.size(2) // 10))
            if audio.size(2) - sz > 0:
                st = random.randint(0, audio.size(2) - sz)
                audio[:, :, st:st + sz] = 0
        if random.random() < 0.3 and audio.size(1) > 10:
            sz = random.randint(1, max(1, audio.size(1) // 10))
            if audio.size(1) - sz > 0:
                st = random.randint(0, audio.size(1) - sz)
                audio[:, st:st + sz, :] = 0
        return audio

def tta_transform(frames, audio, it):
    """Deterministic, batch-safe TTA: identity / horizontal flip / brightness."""
    if it == 0:
        return frames, audio
    if it % 2 == 1:
        return torch.flip(frames, dims=[-1]), audio
    return torch.clamp(frames * 1.1, -2.5, 2.5), audio

# =====================================================================
# DATASET
# =====================================================================

class ImprovedDeepfakeDataset(Dataset):
    def __init__(self, root_dir, split="train", cache_data=True):
        self.root_dir = root_dir
        self.split = split
        self.is_training = (split == "train")
        self.cache = {} if cache_data else None
        self.video_aug = VideoAugmentation()
        self.audio_aug = AudioAugmentation()

        samples = []
        for label_name in ["REAL", "FAKE"]:
            folder = os.path.join(root_dir, label_name)
            if os.path.exists(folder):
                for ext in ["*.mp4", "*.avi", "*.mov"]:
                    for video_path in glob.glob(os.path.join(folder, ext)):
                        samples.append((video_path, 0 if label_name == "REAL" else 1))
        if len(samples) == 0:
            raise ValueError(f"No videos found in {root_dir}")

        # FIXED SPLIT: sorted + dedicated RNG so train/val/test are always
        # the same disjoint partitions (no leakage between the 3 datasets).
        samples.sort()
        rng = random.Random(Config.SPLIT_SEED)
        real_samples = [s for s in samples if s[1] == 0]
        fake_samples = [s for s in samples if s[1] == 1]
        rng.shuffle(real_samples)
        rng.shuffle(fake_samples)

        r_tr, r_va = int(0.7 * len(real_samples)), int(0.85 * len(real_samples))
        f_tr, f_va = int(0.7 * len(fake_samples)), int(0.85 * len(fake_samples))

        if split == "train":
            self.samples = real_samples[:r_tr] + fake_samples[:f_tr]
        elif split == "val":
            self.samples = real_samples[r_tr:r_va] + fake_samples[f_tr:f_va]
        else:
            self.samples = real_samples[r_va:] + fake_samples[f_va:]
        rng.shuffle(self.samples)

        print(f"\n{split} set: {len(self.samples)} videos")
        print(f"  Real: {sum(1 for s in self.samples if s[1] == 0)}")
        print(f"  Fake: {sum(1 for s in self.samples if s[1] == 1)}")

        if cache_data and self.cache is not None:
            self._cache_data()

    def _cache_data(self):
        max_cache = min(Config.MAX_CACHE_SIZE, len(self.samples))
        print(f"\nCaching {max_cache} videos to RAM (float16)...")
        for i in tqdm(range(max_cache), desc="Caching"):
            video_path, _ = self.samples[i]
            try:
                frames = self._extract_frames(video_path).half()   # half the RAM
                audio = self._extract_audio(video_path)
                self.cache[video_path] = (frames, audio)
            except Exception:
                pass
        print(f"Cached {len(self.cache)} videos")

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return torch.zeros(Config.NUM_FRAMES, 3, Config.FRAME_SIZE, Config.FRAME_SIZE)
        indices = np.linspace(0, max(0, total_frames - 1), Config.NUM_FRAMES, dtype=int)
        raw_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (Config.FRAME_SIZE, Config.FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
                raw_frames.append(frame)
            elif raw_frames:
                raw_frames.append(raw_frames[-1])
        cap.release()
        while len(raw_frames) < Config.NUM_FRAMES:
            raw_frames.append(raw_frames[-1] if raw_frames else np.zeros((Config.FRAME_SIZE, Config.FRAME_SIZE, 3), dtype=np.uint8))
        arr = np.stack(raw_frames[:Config.NUM_FRAMES]).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(0, 3, 1, 2)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (t - mean) / std

    def _extract_audio(self, video_path):
        try:
            audio, sr = librosa.load(video_path, sr=Config.AUDIO_SR, duration=Config.AUDIO_DURATION,
                                     res_type='kaiser_fast', mono=True)
            target = int(Config.AUDIO_SR * Config.AUDIO_DURATION)
            audio = np.pad(audio, (0, target - len(audio))) if len(audio) < target else audio[:target]
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=512,
                                                 n_fft=1024, window='hann', center=False)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_n = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
            return torch.FloatTensor(mel_n).unsqueeze(0)
        except Exception:
            return torch.zeros(1, 128, audio_time_frames())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        if self.cache is not None and video_path in self.cache:
            frames, audio = self.cache[video_path]
            frames = frames.float()          # new tensor (cache stays untouched)
            audio = audio.clone()
        else:
            try:
                frames = self._extract_frames(video_path)
                audio = self._extract_audio(video_path)
            except Exception:
                frames = torch.zeros(Config.NUM_FRAMES, 3, Config.FRAME_SIZE, Config.FRAME_SIZE)
                audio = torch.zeros(1, 128, audio_time_frames())

        if self.is_training and Config.USE_AUGMENTATION:
            frames = self.video_aug.apply_spatial_aug(frames, Config.AUGMENTATION_PROB)
            frames = self.video_aug.apply_temporal_aug(frames, Config.AUGMENTATION_PROB * 0.6)
            audio = self.audio_aug.apply_audio_aug(audio, Config.AUGMENTATION_PROB)
        return frames, audio, torch.tensor(label, dtype=torch.long)

# =====================================================================
# MODELS (unchanged architecture)
# =====================================================================

class ImprovedVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        for param in list(self.feature_extractor.parameters())[:60]:
            param.requires_grad = False
        self.temporal_lstm = nn.LSTM(2048, 512, 2, batch_first=True, dropout=0.4, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(1024, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(256),
            nn.Linear(256, 2))

    def forward(self, frames):
        b, t = frames.shape[:2]
        x = self.feature_extractor(frames.reshape(b * t, *frames.shape[2:]))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(b, t, -1)
        lstm_out, _ = self.temporal_lstm(x)
        w = F.softmax(self.attention(lstm_out), dim=1)
        return self.classifier((lstm_out * w).sum(dim=1))

class ImprovedAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(128),
            nn.Linear(128, 2))

    def forward(self, audio):
        x = self.conv_layers(audio)
        return self.classifier(x.view(x.size(0), -1))

class ImprovedFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.video_features = efficientnet.features
        for param in list(self.video_features.parameters())[:30]:
            param.requires_grad = False
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.fusion = nn.Sequential(
            nn.Linear(1280 + 128, 512), nn.ReLU(), nn.Dropout(0.5), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(256),
            nn.Linear(256, 2))

    def forward(self, frames, audio):
        mid = frames[:, frames.size(1) // 2]
        v = F.adaptive_avg_pool2d(self.video_features(mid), (1, 1)).view(mid.size(0), -1)
        a = self.audio_conv(audio).view(audio.size(0), -1)
        return self.fusion(torch.cat([v, a], dim=1))

class ImprovedEnsembleDetector(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.video_model = ImprovedVideoModel()
        self.audio_model = ImprovedAudioModel()
        self.fusion_model = ImprovedFusionModel()
        if weights is None:
            weights = Config.ENSEMBLE_WEIGHTS
        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, frames, audio):
        vp = self.video_model(frames)
        ap = self.audio_model(audio)
        fp = self.fusion_model(frames, audio)
        prob = (self.weights[0] * F.softmax(vp, 1) + self.weights[1] * F.softmax(ap, 1)
                + self.weights[2] * F.softmax(fp, 1))
        return torch.log(prob + 1e-8), {'video': vp, 'audio': ap, 'fusion': fp}

# =====================================================================
# >>> METRICS_BEGIN
# =====================================================================

EPS = 1e-12
CLASSES = ['Real', 'Fake']

def _confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn

def quick_metrics(tp, tn, fp, fn):
    """Fast threshold-dependent metrics straight from the confusion matrix."""
    n = tp + tn + fp + fn
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    spec = tn / (tn + fp + EPS)
    denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    po = (tp + tn) / max(n, 1)
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / max(n, 1) ** 2
    kappa = (po - pe) / (1 - pe + EPS)
    return {
        'Accuracy': po, 'Balanced Acc.': (rec + spec) / 2, 'Precision': prec, 'Recall': rec,
        'Specificity': spec, 'F1-Score': 2 * prec * rec / (prec + rec + EPS),
        'MCC': mcc, "Cohen's Kappa": kappa,
    }

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    m = quick_metrics(tp, tn, fp, fn)
    prec, rec, spec = m['Precision'], m['Recall'], m['Specificity']
    npv = tn / (tn + fn + EPS)
    f1_real = 2 * tn / (2 * tn + fn + fp + EPS)
    sup_r, sup_f = tn + fp, tp + fn
    m.update({
        'NPV': npv,
        'F2-Score': 5 * prec * rec / (4 * prec + rec + EPS),
        'Macro-F1': (m['F1-Score'] + f1_real) / 2,
        'Weighted-F1': (m['F1-Score'] * sup_f + f1_real * sup_r) / max(sup_f + sup_r, 1),
        'G-Mean': math.sqrt(rec * spec),
        "Youden's J": rec + spec - 1,
        'Markedness': prec + npv - 1,
        'Fowlkes-Mallows': math.sqrt(prec * rec),
        'Jaccard': tp / (tp + fp + fn + EPS),
        'FPR': 1 - spec, 'FNR': 1 - rec, 'FDR': 1 - prec, 'FOR': 1 - npv,
        'Error Rate': 1 - m['Accuracy'],
        'LR+': rec / ((1 - spec) + EPS), 'LR-': (1 - rec) / (spec + EPS),
    })
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        pc = np.clip(y_prob, 1e-7, 1 - 1e-7)
        m.update({
            'AUC-ROC': roc_auc_score(y_true, y_prob),
            'AP (AUC-PR)': average_precision_score(y_true, y_prob),
            'pAUC@FPR<=0.1': roc_auc_score(y_true, y_prob, max_fpr=0.1),
            'EER': (fpr[i] + fnr[i]) / 2,
            'KS Statistic': float(np.max(tpr - fpr)),
            'TPR@FPR=1%': float(np.interp(0.01, fpr, tpr)),
            'TPR@FPR=0.1%': float(np.interp(0.001, fpr, tpr)),
            'Log Loss': float(-np.mean(y_true * np.log(pc) + (1 - y_true) * np.log(1 - pc))),
            'Brier Score': float(np.mean((y_prob - y_true) ** 2)),
            'ECE': expected_calibration_error(y_true, y_prob),
        })
    m.update({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})
    return {k: float(v) for k, v in m.items()}

BOOT_KEYS = ['Accuracy', 'Balanced Acc.', 'Precision', 'Recall', 'Specificity',
             'F1-Score', 'MCC', "Cohen's Kappa", 'AUC-ROC']

def bootstrap_metrics(y_true, y_prob, n_iter=None, threshold=0.5, seed=Config.SEED):
    """Non-parametric bootstrap -> (distributions, 95% CIs)."""
    n_iter = n_iter or Config.BOOTSTRAP_ITERS
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)
    store = {k: [] for k in BOOT_KEYS}
    for _ in range(n_iter):
        idx = rng.integers(0, N, N)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.min() == yt.max():
            continue
        q = quick_metrics(*_confusion(yt, (yp >= threshold).astype(int)))
        for k in BOOT_KEYS[:-1]:
            store[k].append(q[k])
        store['AUC-ROC'].append(roc_auc_score(yt, yp))
    dist = {k: np.array(v) for k, v in store.items()}
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in dist.items() if len(v)}
    return dist, ci

# ---------------------------------------------------------------------
# Efficiency / timing utilities
# ---------------------------------------------------------------------

def time_fn(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    cuda_sync()
    out = []
    for _ in range(repeats):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        out.append(time.perf_counter() - t0)
    return np.array(out)

def count_gflops(fn):
    """FLOPs (=2xMACs) via torch FlopCounterMode. NB: LSTM ops are not counted by PyTorch."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops() / 1e9
    except Exception:
        return float('nan')

@torch.no_grad()
def run_efficiency_benchmark(model, loader, repeats=20, warmup=5, single_repeats=100):
    dev = Config.DEVICE
    model.eval()
    frames, audio, _ = next(iter(loader))
    frames, audio = frames.to(dev), audio.to(dev)

    def take(bs):
        reps = math.ceil(bs / frames.size(0))
        return frames.repeat(reps, 1, 1, 1, 1)[:bs], audio.repeat(reps, 1, 1, 1)[:bs]

    comps = {
        'Video': lambda f, a: model.video_model(f),
        'Audio': lambda f, a: model.audio_model(a),
        'Fusion': lambda f, a: model.fusion_model(f, a),
        'Ensemble': lambda f, a: model(f, a),
    }
    mods = {'Video': model.video_model, 'Audio': model.audio_model,
            'Fusion': model.fusion_model, 'Ensemble': model}

    def run(fn, f, a):
        with amp_ctx():
            fn(f, a)

    out = {'single_ms': {}, 'sweep': {}, 'complexity': {}}
    f1, a1 = take(1)
    for name, fn in comps.items():
        out['single_ms'][name] = time_fn(lambda fn=fn: run(fn, f1, a1), single_repeats, warmup) * 1000

    if Config.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for bs in Config.BENCH_BATCH_SIZES:
        try:
            f, a = take(bs)
            t = time_fn(lambda: run(comps['Ensemble'], f, a), repeats, warmup)
            out['sweep'][bs] = {
                'mean_ms': float(t.mean() * 1000), 'std_ms': float(t.std() * 1000),
                'ms_per_video': float(t.mean() * 1000 / bs), 'std_ms_per_video': float(t.std() * 1000 / bs),
                'throughput_vps': float(bs / t.mean()), 'throughput_fps': float(bs * Config.NUM_FRAMES / t.mean()),
            }
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    out['peak_gpu_mem_MB'] = (torch.cuda.max_memory_allocated() / 1e6) if Config.DEVICE == "cuda" else float('nan')

    for name, mod in mods.items():
        n_tot = sum(p.numel() for p in mod.parameters())
        n_tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        size_mb = (sum(p.numel() * p.element_size() for p in mod.parameters())
                   + sum(b.numel() * b.element_size() for b in mod.buffers())) / 1e6
        out['complexity'][name] = {
            'params_M': n_tot / 1e6, 'trainable_M': n_tr / 1e6, 'size_MB': size_mb,
            'GFLOPs': count_gflops(lambda fn=comps[name]: fn(f1, a1)),
        }
    return out

def fmt_time(s):
    if s >= 3600: return f"{s / 3600:.2f} h"
    if s >= 60: return f"{s / 60:.1f} min"
    return f"{s:.2f} s"

def to_serializable(o):
    if isinstance(o, dict): return {str(k): to_serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_serializable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
# <<< METRICS_END

# =====================================================================
# >>> STATS_BEGIN
# Reproducibility / significance / thresholds / ensemble weights
# (pure numpy + scipy + sklearn: works on saved probabilities, no GPU needed)
# =====================================================================
from sklearn.linear_model import LogisticRegression

COMPONENTS = ['Video', 'Audio', 'Fusion']

# ---------------------------------------------------------------------
# 1) Reproducibility across seeds
# ---------------------------------------------------------------------
REPRO_KEYS = ['Accuracy', 'Balanced Acc.', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC',
              "Cohen's Kappa", 'AUC-ROC', 'AP (AUC-PR)', 'EER', 'Log Loss', 'Brier Score', 'ECE']

def t_confidence_interval(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float('nan'), float('nan'))
    h = stats.sem(x) * stats.t.ppf((1 + conf) / 2, len(x) - 1)
    return (float(x.mean() - h), float(x.mean() + h))

def per_seed_table(per_seed):
    """Long table: one row per (seed, model) with every metric."""
    rows = []
    for seed, S in per_seed.items():
        for model_name, m in S['metrics'].items():
            rows.append({'seed': seed, 'model': model_name, **m})
    return pd.DataFrame(rows)

def reproducibility_summary(df_seed, keys=REPRO_KEYS):
    """mean / std / 95% CI / min / max / CV% per model and metric across seeds."""
    rows = []
    for model_name, g in df_seed.groupby('model', sort=False):
        for k in keys:
            if k not in g:
                continue
            v = g[k].to_numpy(dtype=float)
            lo, hi = t_confidence_interval(v)
            mean = float(np.mean(v))
            std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            rows.append({'model': model_name, 'metric': k, 'n_seeds': len(v), 'mean': mean, 'std': std,
                         'ci95_low': lo, 'ci95_high': hi, 'min': float(v.min()), 'max': float(v.max()),
                         'range': float(v.max() - v.min()),
                         'cv_percent': float(abs(std / mean) * 100) if mean != 0 else float('nan'),
                         'mean_pm_std': f"{mean:.4f} ± {std:.4f}"})
    return pd.DataFrame(rows)

def repeatability_check(p_runs):
    """p_runs: list of probability vectors from repeated runs of the SAME seed / same model."""
    base = np.asarray(p_runs[0])
    rows = []
    for i, p in enumerate(p_runs[1:], start=2):
        d = np.abs(np.asarray(p) - base)
        rows.append({'run': i, 'max_abs_prob_diff': float(d.max()), 'mean_abs_prob_diff': float(d.mean()),
                     'identical_predictions_%': float(((np.asarray(p) >= 0.5) == (base >= 0.5)).mean() * 100)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# 2) Significance tests
# ---------------------------------------------------------------------
def mcnemar_test(y, p_a, p_b, thr=0.5):
    """Exact McNemar test on paired correctness (two-sided)."""
    y = np.asarray(y).astype(int)
    ca = ((np.asarray(p_a) >= thr).astype(int) == y)
    cb = ((np.asarray(p_b) >= thr).astype(int) == y)
    b = int((ca & ~cb).sum())      # A right, B wrong
    c = int((~ca & cb).sum())      # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5).pvalue)
    return {'A_right_B_wrong': b, 'A_wrong_B_right': c, 'p_value': p}

def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def delong_test(y, p_a, p_b):
    """DeLong test for two correlated ROC AUCs (Sun & Xu, 2014 fast implementation)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind='mergesort')          # positives first
    m = int(y.sum()); n = len(y) - m
    if m < 2 or n < 2:
        return {'auc_A': float('nan'), 'auc_B': float('nan'), 'z': float('nan'), 'p_value': float('nan')}
    preds = np.vstack([np.asarray(p_a)[order], np.asarray(p_b)[order]]).astype(float)
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 1e-18:
        return {'auc_A': float(aucs[0]), 'auc_B': float(aucs[1]), 'z': float('nan'),
                'p_value': 1.0 if abs(aucs[0] - aucs[1]) < 1e-12 else 0.0}
    z = (aucs[0] - aucs[1]) / math.sqrt(var)
    return {'auc_A': float(aucs[0]), 'auc_B': float(aucs[1]), 'z': float(z), 'p_value': float(2 * stats.norm.sf(abs(z)))}

def paired_bootstrap_diff(y, p_a, p_b, thr=0.5, n_iter=None, seed=Config.SEED):
    """Paired bootstrap of (A - B) for Accuracy and F1 -> mean diff, 95% CI, two-sided p."""
    n_iter = n_iter or Config.BOOTSTRAP_ITERS
    y = np.asarray(y).astype(int)
    pa, pb = (np.asarray(p_a) >= thr).astype(int), (np.asarray(p_b) >= thr).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_iter, len(y)))
    yy = y[idx]

    def acc(pred):
        return (pred[idx] == yy).mean(1)

    def f1(pred):
        pp = pred[idx]
        tp = (pp * yy).sum(1); fp = (pp * (1 - yy)).sum(1); fn = ((1 - pp) * yy).sum(1)
        return 2 * tp / (2 * tp + fp + fn + EPS)

    out = {}
    for name, fn in [('Accuracy', acc), ('F1-Score', f1)]:
        d = fn(pa) - fn(pb)
        p = float(np.clip(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (n_iter + 1), 1.0))   # floor = resolution of the bootstrap
        out[name] = {'ci_low': float(np.percentile(d, 2.5)), 'ci_high': float(np.percentile(d, 97.5)),
                     'p_value': float(p)}
    return out

def holm_correction(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    order = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out

def sig_stars(p):
    if not np.isfinite(p): return 'n/a'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

def significance_within_seed(y, probs, ref='Ensemble', thr=0.5, seed=None):
    """Ensemble vs every other model on the SAME test set: McNemar, DeLong and paired bootstrap."""
    rows = []
    for other in [m for m in probs if m != ref]:
        mc = mcnemar_test(y, probs[ref], probs[other], thr)
        dl = delong_test(y, probs[ref], probs[other])
        pa, pb = (probs[ref] >= thr).astype(int), (probs[other] >= thr).astype(int)
        bs = paired_bootstrap_diff(y, probs[ref], probs[other], thr)
        acc_diff = float(np.mean(pa == y) - np.mean(pb == y))
        f1_diff = float(quick_metrics(*_confusion(y, pa))['F1-Score'] - quick_metrics(*_confusion(y, pb))['F1-Score'])
        rows.append({'seed': seed, 'A': ref, 'B': other,
                     'acc_A': float(np.mean(pa == y)), 'acc_B': float(np.mean(pb == y)), 'acc_diff': acc_diff,
                     'acc_diff_ci_low': bs['Accuracy']['ci_low'], 'acc_diff_ci_high': bs['Accuracy']['ci_high'],
                     'bootstrap_p_acc': bs['Accuracy']['p_value'],
                     'f1_diff': f1_diff, 'f1_diff_ci_low': bs['F1-Score']['ci_low'],
                     'f1_diff_ci_high': bs['F1-Score']['ci_high'], 'bootstrap_p_f1': bs['F1-Score']['p_value'],
                     'mcnemar_A_right_B_wrong': mc['A_right_B_wrong'], 'mcnemar_A_wrong_B_right': mc['A_wrong_B_right'],
                     'mcnemar_p': mc['p_value'], 'auc_A': dl['auc_A'], 'auc_B': dl['auc_B'],
                     'auc_diff': dl['auc_A'] - dl['auc_B'], 'delong_z': dl['z'], 'delong_p': dl['p_value']})
    df = pd.DataFrame(rows)
    df['mcnemar_sig'] = df['mcnemar_p'].map(sig_stars)
    df['delong_sig'] = df['delong_p'].map(sig_stars)
    return df

def pairwise_mcnemar_matrix(y, probs, thr=0.5):
    names = list(probs)
    M = np.ones((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                M[i, j] = mcnemar_test(y, probs[a], probs[b], thr)['p_value']
    return names, M

SIG_METRICS = ['Accuracy', 'F1-Score', 'AUC-ROC', 'MCC', "Cohen's Kappa"]

def significance_across_seeds(df_seed, ref='Ensemble', metrics=SIG_METRICS, target=0.90):
    """Paired t-test + Wilcoxon over seeds (Ensemble vs each model) with Holm correction,
    plus a one-sided t-test of the Ensemble against the 90% target."""
    rows = []
    seeds = sorted(df_seed['seed'].unique())
    for k in metrics:
        a = df_seed[df_seed.model == ref].set_index('seed').loc[seeds, k].to_numpy(float)
        for other in [m for m in df_seed.model.unique() if m != ref]:
            b = df_seed[df_seed.model == other].set_index('seed').loc[seeds, k].to_numpy(float)
            d = a - b
            sd = d.std(ddof=1) if len(d) > 1 else float('nan')
            if len(d) > 1 and sd > 0:
                t, p_t = stats.ttest_rel(a, b)
            else:
                t, p_t = float('nan'), (1.0 if np.allclose(d, 0) else 0.0)
            try:
                w, p_w = stats.wilcoxon(a, b)
            except Exception:
                w, p_w = float('nan'), float('nan')
            lo, hi = t_confidence_interval(d)
            rows.append({'metric': k, 'A': ref, 'B': other, 'n_seeds': len(d), 'mean_A': a.mean(), 'mean_B': b.mean(),
                         'mean_diff': d.mean(), 'diff_ci95_low': lo, 'diff_ci95_high': hi,
                         'cohens_dz': float(d.mean() / sd) if (len(d) > 1 and sd > 0) else float('nan'),
                         't_stat': float(t), 'paired_t_p': float(p_t), 'wilcoxon_p': float(p_w)})
    df = pd.DataFrame(rows)
    df['paired_t_p_holm'] = holm_correction(df['paired_t_p'].to_numpy())
    df['t_sig_holm'] = df['paired_t_p_holm'].map(sig_stars)
    n = len(seeds)
    df['note'] = (f"n={n} seeds: Wilcoxon cannot go below p={2 / 2 ** n:.4f}; interpret with the effect size / CI"
                  if n < 6 else '')
    # one-sided test against the target (is mean accuracy > 90% ?)
    tgt = []
    for k in ['Accuracy', 'F1-Score', 'AUC-ROC']:
        a = df_seed[df_seed.model == ref].set_index('seed').loc[seeds, k].to_numpy(float)
        if len(a) > 1 and a.std(ddof=1) > 0:
            t, p = stats.ttest_1samp(a, target, alternative='greater')
        else:
            t, p = float('nan'), float('nan')
        tgt.append({'metric': k, 'ensemble_mean': a.mean(), 'target': target, 't_stat': float(t), 'one_sided_p_greater_than_target': float(p),
                    'ci95_low': t_confidence_interval(a)[0], 'ci95_high': t_confidence_interval(a)[1]})
    return df, pd.DataFrame(tgt)

# ---------------------------------------------------------------------
# 3) Four decision thresholds (selected on VALIDATION, applied to TEST)
# ---------------------------------------------------------------------
THRESHOLD_LABELS = {
    'default_0.5': 'Default (0.5)',
    'youden_val': "Youden's J (val)",
    'max_f1_val': 'Max-F1 (val)',
    'eer_val': 'EER (val)',
}

def select_thresholds(y_val, p_val, strategies=None):
    strategies = strategies or Config.THRESHOLD_STRATEGIES
    y_val = np.asarray(y_val).astype(int); p_val = np.asarray(p_val, float)
    out = {}
    fpr, tpr, thr = roc_curve(y_val, p_val)
    finite = np.isfinite(thr)
    for s in strategies:
        if s == 'default_0.5':
            out[s] = 0.5
        elif s == 'youden_val':
            j = np.where(finite, tpr - fpr, -np.inf)
            out[s] = float(np.clip(thr[int(np.argmax(j))], 0, 1))
        elif s == 'max_f1_val':
            pr, rc, th = precision_recall_curve(y_val, p_val)
            f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + EPS)
            out[s] = float(th[int(np.argmax(f1))])
        elif s == 'eer_val':
            fnr = 1 - tpr
            d = np.where(finite, np.abs(fnr - fpr), np.inf)
            out[s] = float(np.clip(thr[int(np.argmin(d))], 0, 1))
        elif s.startswith('fixed_'):
            out[s] = float(s.split('_')[1])
    return out

def threshold_comparison(y_val, p_val, y_test, p_test, strategies=None, seed=None):
    th = select_thresholds(y_val, p_val, strategies)
    rows = []
    for name, t in th.items():
        m = compute_all_metrics(y_test, p_test, threshold=t)
        rows.append({'seed': seed, 'strategy': name, 'label': THRESHOLD_LABELS.get(name, name), 'threshold': t, **m})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# 4) Ensemble-weight comparison (post-hoc re-weighting of the stored component probabilities)
#    The ensemble probability is LINEAR in the component probabilities, so this is exact.
# ---------------------------------------------------------------------
def ens_from_weights(comp, w):
    w = np.asarray(w, float); w = w / w.sum()
    return w[0] * comp['Video'] + w[1] * comp['Audio'] + w[2] * comp['Fusion']

def simplex_grid(step=0.05):
    n = int(round(1 / step))
    return [(a / n, b / n, (n - a - b) / n) for a in range(n + 1) for b in range(n + 1 - a)]

def weight_comparison(comp_val, y_val, comp_test, y_test, seed=None, configs=None, grid_step=0.05):
    """comp_*: dict Video/Audio/Fusion -> P(fake).  Returns (table, simplex-grid table)."""
    configs = dict(configs or Config.WEIGHT_CONFIGS)
    y_val = np.asarray(y_val).astype(int); y_test = np.asarray(y_test).astype(int)

    # val-optimised weights (grid search on the simplex, maximise val AUC, tie-break val accuracy)
    grid = simplex_grid(grid_step)
    g_rows = []
    for w in grid:
        pv, pt = ens_from_weights(comp_val, w), ens_from_weights(comp_test, w)
        g_rows.append({'seed': seed, 'w_video': w[0], 'w_audio': w[1], 'w_fusion': w[2],
                       'val_auc': roc_auc_score(y_val, pv), 'val_acc': float(((pv >= .5) == y_val).mean()),
                       'test_auc': roc_auc_score(y_test, pt), 'test_acc': float(((pt >= .5) == y_test).mean())})
    gdf = pd.DataFrame(g_rows)
    best = gdf.sort_values(['val_auc', 'val_acc'], ascending=False).iloc[0]
    configs['Val-optimised'] = [best.w_video, best.w_audio, best.w_fusion]

    # AUC-proportional weights
    aucs = np.array([roc_auc_score(y_val, comp_val[c]) for c in COMPONENTS])
    configs['Val-AUC-proportional'] = list(np.clip(aucs - 0.5, 1e-6, None))

    rows, probs = [], {}
    for name, w in configs.items():
        wn = np.asarray(w, float) / np.sum(w)
        p = ens_from_weights(comp_test, wn)
        probs[name] = p
        m = compute_all_metrics(y_test, p)
        rows.append({'seed': seed, 'config': name, 'w_video': wn[0], 'w_audio': wn[1], 'w_fusion': wn[2], **m})

    # stacked logistic regression fitted on validation probabilities
    def feats(c):
        return np.column_stack([np.log(np.clip(c[k], 1e-6, 1 - 1e-6) / (1 - np.clip(c[k], 1e-6, 1 - 1e-6))) for k in COMPONENTS])
    try:
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats(comp_val), y_val)
        p = lr.predict_proba(feats(comp_test))[:, 1]
        probs['Stacked-LR'] = p
        m = compute_all_metrics(y_test, p)
        rows.append({'seed': seed, 'config': 'Stacked-LR', 'w_video': np.nan, 'w_audio': np.nan, 'w_fusion': np.nan, **m})
    except Exception as e:
        print(f"  (stacked LR skipped: {e})")

    df = pd.DataFrame(rows)
    ref = 'Default'
    if ref in probs:
        df['mcnemar_p_vs_default'] = [mcnemar_test(y_test, probs[c], probs[ref])['p_value'] if c != ref else np.nan
                                      for c in df.config]
        df['delong_p_vs_default'] = [delong_test(y_test, probs[c], probs[ref])['p_value'] if c != ref else np.nan
                                     for c in df.config]
    return df, gdf, probs

def aggregate_over_seeds(df, group_col, keys):
    """mean ± std over seeds for every value of group_col."""
    rows = []
    for g, sub in df.groupby(group_col, sort=False):
        r = {group_col: g}
        for k in keys:
            if k in sub:
                v = sub[k].to_numpy(float)
                r[f'{k}_mean'] = v.mean()
                r[f'{k}_std'] = v.std(ddof=1) if len(v) > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)
# <<< STATS_END

# =====================================================================
# >>> PLOTTER_BEGIN
# =====================================================================

# --- publication style (Okabe-Ito colour-blind-safe palette) ---
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00',
     'purple': '#CC79A7', 'sky': '#56B4E9', 'grey': '#7F7F7F', 'black': '#000000'}
MODEL_COLORS = {'Video': C['blue'], 'Audio': C['orange'], 'Fusion': C['green'], 'Ensemble': C['red']}
MODELS = ['Video', 'Audio', 'Fusion', 'Ensemble']

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'axes.linewidth': 1.3, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'grid.alpha': 0.3, 'grid.linestyle': '--', 'lines.linewidth': 2.4,
    'figure.dpi': 110, 'savefig.dpi': 600, 'savefig.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

SQ, WIDE = (6.4, 5.4), (8.6, 5.4)

def _display(fig):
    if not Config.SHOW_PLOTS:
        return
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(fig)
            return
    except Exception:
        pass
    plt.show()

class _FigBase:
    """Every figure: build -> DISPLAY -> SAVE (600 dpi PNG + vector PDF) -> close."""

    def __init__(self, save_dir, prefix=''):
        self.dir = Path(save_dir) / f"figures_{Config.DPI}dpi"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.n = prefix, 0

    def _finish(self, fig, slug):
        self.n += 1
        name = f"{self.prefix}{self.n:02d}_{slug}"
        fig.tight_layout()
        fig.canvas.draw()
        _display(fig)                                               # 1) display
        fig.savefig(self.dir / f"{name}.png", dpi=Config.DPI, bbox_inches='tight', facecolor='white')  # 2) save
        if Config.SAVE_PDF:
            fig.savefig(self.dir / f"{name}.pdf", bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ saved {name}.png @ {Config.DPI} dpi")

    @staticmethod
    def _style(ax, xl=None, yl=None, title=None, grid=True):
        if xl: ax.set_xlabel(xl, fontweight='bold')
        if yl: ax.set_ylabel(yl, fontweight='bold')
        if title and Config.SHOW_TITLES: ax.set_title(title, fontweight='bold', pad=12)
        if grid: ax.grid(True)

    @staticmethod
    def _bar_labels(ax, bars, fmt='{:.3f}', fs=10, pad=0.01):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h), ha='center', va='bottom',
                        fontsize=fs, fontweight='bold')

    def _table_fig(self, df, slug, title, colw=1.9):
        fig, ax = plt.subplots(figsize=(2.6 + colw * df.shape[1], 0.5 * (len(df) + 2) + 1))
        ax.axis('off')
        tb = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc='center', loc='center')
        tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.55)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor('#BBBBBB')
            if r == 0:
                cell.set_facecolor(C['blue']); cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            if c == -1:
                cell.set_text_props(fontweight='bold')
        if Config.SHOW_TITLES: ax.set_title(title, fontweight='bold', pad=8)
        self._finish(fig, slug)


class ResearchPlotter(_FigBase):
    """Primary-seed deep dive: classification, training, efficiency and table figures."""

    def __init__(self, save_dir, R):
        super().__init__(save_dir, '')
        self.R = R
        self.y = np.asarray(R['y_true']).astype(int)
        self.p = np.asarray(R['probs']['Ensemble'])
        self.pred = (self.p >= 0.5).astype(int)
        self.m = R['metrics']['Ensemble']

    def _ci_str(self, k):
        ci = self.R['ci'].get(k)
        return f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""

    # ---------- classification-performance figures ----------
    def confusion_counts(self):
        cm = confusion_matrix(self.y, self.pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=SQ)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, square=True, linewidths=1.5, linecolor='white',
                    xticklabels=CLASSES, yticklabels=CLASSES, cbar_kws={'label': 'Number of videos'},
                    annot_kws={'size': 20, 'weight': 'bold'})
        self._style(ax, 'Predicted label', 'True label', 'Confusion Matrix (counts)', grid=False)
        self._finish(fig, 'confusion_matrix_counts')

    def confusion_normalized(self):
        cm = confusion_matrix(self.y, self.pred, labels=[0, 1])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{cmn[i, j] * 100:.2f}%\n(n={cm[i, j]})" for j in range(2)] for i in range(2)])
        fig, ax = plt.subplots(figsize=SQ)
        sns.heatmap(cmn, annot=annot, fmt='', cmap='Blues', vmin=0, vmax=1, ax=ax, square=True, linewidths=1.5,
                    linecolor='white', xticklabels=CLASSES, yticklabels=CLASSES,
                    cbar_kws={'label': 'Row-normalised rate'}, annot_kws={'size': 15, 'weight': 'bold'})
        self._style(ax, 'Predicted label', 'True label', 'Normalised Confusion Matrix', grid=False)
        self._finish(fig, 'confusion_matrix_normalized')

    def _roc_band(self, n=300):
        rng = np.random.default_rng(Config.SEED)
        grid = np.linspace(0, 1, 501)
        tprs = []
        for _ in range(n):
            idx = rng.integers(0, len(self.y), len(self.y))
            if self.y[idx].min() == self.y[idx].max():
                continue
            f, t, _ = roc_curve(self.y[idx], self.p[idx])
            tprs.append(np.interp(grid, f, t))
        lo, hi = np.percentile(tprs, [2.5, 97.5], axis=0)
        return grid, lo, hi

    def roc(self):
        fpr, tpr, thr = roc_curve(self.y, self.p)
        grid, lo, hi = self._roc_band()
        fig, ax = plt.subplots(figsize=SQ)
        ax.plot(fpr, tpr, color=C['blue'], label=f"Ensemble (AUC = {self.m['AUC-ROC']:.4f})")
        ax.fill_between(grid, lo, hi, color=C['blue'], alpha=0.2, label='95% bootstrap CI')
        ax.plot([0, 1], [0, 1], '--', color=C['grey'], lw=1.5, label='Chance')
        e = self.m['EER']
        ax.plot(e, 1 - e, 'o', ms=10, color=C['orange'], mec='k', label=f'EER = {e:.4f}')
        j = int(np.argmax(tpr - fpr))
        ax.plot(fpr[j], tpr[j], 's', ms=10, color=C['green'], mec='k', label=f"Youden's J (thr = {thr[j]:.3f})")
        self._style(ax, 'False Positive Rate', 'True Positive Rate', 'Receiver Operating Characteristic')
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
        ax.legend(loc='lower right', frameon=True)
        self._finish(fig, 'roc_curve')

    def roc_semilog(self):
        fig, ax = plt.subplots(figsize=SQ)
        for n in MODELS:
            f, t, _ = roc_curve(self.y, self.R['probs'][n])
            k = f > 0
            ax.plot(f[k], t[k], color=MODEL_COLORS[n], label=f"{n} (AUC = {self.R['metrics'][n]['AUC-ROC']:.4f})",
                    lw=3 if n == 'Ensemble' else 2)
        ax.set_xscale('log'); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1.02)
        self._style(ax, 'False Positive Rate (log scale)', 'True Positive Rate', 'ROC Curves - Log-scale FPR (all models)')
        ax.legend(loc='lower right')
        self._finish(fig, 'roc_all_models_semilog')

    def det(self):
        fig, ax = plt.subplots(figsize=SQ)
        for n in MODELS:
            f, fn, _ = det_curve(self.y, self.R['probs'][n])
            ax.plot(stats.norm.ppf(np.clip(f, 1e-4, 1 - 1e-4)), stats.norm.ppf(np.clip(fn, 1e-4, 1 - 1e-4)),
                    color=MODEL_COLORS[n], label=f"{n} (EER = {self.R['metrics'][n]['EER'] * 100:.2f}%)",
                    lw=3 if n == 'Ensemble' else 2)
        ticks = np.array([0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95])
        pos = stats.norm.ppf(ticks)
        lab = [f"{t * 100:g}" for t in ticks]
        ax.set_xticks(pos); ax.set_xticklabels(lab); ax.set_yticks(pos); ax.set_yticklabels(lab)
        lim = [stats.norm.ppf(0.0007), stats.norm.ppf(0.96)]
        ax.plot(lim, lim, ':', color=C['grey'], label='EER line')
        ax.set_xlim(lim); ax.set_ylim(lim)
        self._style(ax, 'False Positive Rate (%)', 'False Negative Rate (%)', 'Detection Error Tradeoff (DET)')
        ax.legend(loc='upper right')
        self._finish(fig, 'det_curve')

    def pr(self):
        pr, rc, _ = precision_recall_curve(self.y, self.p)
        f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + EPS)
        b = int(np.argmax(f1))
        fig, ax = plt.subplots(figsize=SQ)
        ax.step(rc, pr, where='post', color=C['blue'], label=f"Ensemble (AP = {self.m['AP (AUC-PR)']:.4f})")
        ax.axhline(self.y.mean(), ls='--', color=C['grey'], label=f'Prevalence = {self.y.mean():.3f}')
        ax.plot(rc[b], pr[b], 'o', ms=10, color=C['red'], mec='k', label=f'Best F1 = {f1[b]:.4f}')
        self._style(ax, 'Recall', 'Precision', 'Precision-Recall Curve')
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.03)
        ax.legend(loc='lower left')
        self._finish(fig, 'precision_recall_curve')

    def calibration(self, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(self.p, bins) - 1, 0, n_bins - 1)
        conf, acc, cnt = [], [], []
        for b in range(n_bins):
            m = idx == b
            if m.any():
                conf.append(self.p[m].mean()); acc.append(self.y[m].mean()); cnt.append(m.sum())
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax.plot([0, 1], [0, 1], '--', color=C['grey'], label='Perfectly calibrated')
        ax.plot(conf, acc, 'o-', color=C['blue'], ms=8, label='Ensemble')
        ax.text(0.05, 0.85, f"ECE = {self.m['ECE']:.4f}\nBrier = {self.m['Brier Score']:.4f}\nLog-loss = {self.m['Log Loss']:.4f}",
                transform=ax.transAxes, bbox=dict(boxstyle='round', fc='white', ec=C['grey']))
        self._style(ax, None, 'Fraction of positives', 'Reliability Diagram')
        ax.legend(loc='lower right'); ax.set_ylim(-0.02, 1.02)
        ax2.hist(self.p, bins=bins, color=C['blue'], alpha=0.8, edgecolor='k')
        ax2.set_yscale('log')
        self._style(ax2, 'Mean predicted probability (Fake)', 'Count')
        self._finish(fig, 'calibration_reliability')

    def score_distribution(self):
        fig, ax = plt.subplots(figsize=WIDE)
        for lab, c, nm in [(0, C['green'], 'Real'), (1, C['red'], 'Fake')]:
            s = self.p[self.y == lab]
            ax.hist(s, bins=40, range=(0, 1), density=True, alpha=0.55, color=c, edgecolor='k', lw=0.6, label=f'{nm} (n={len(s)})')
            if len(s) > 2 and np.std(s) > 1e-6:
                sns.kdeplot(s, ax=ax, color=c, lw=2.2, clip=(0, 1), bw_adjust=0.8)
        ax.axvline(0.5, ls='--', color='k', label='Threshold = 0.5')
        self._style(ax, 'Predicted probability of Fake', 'Density', 'Score Distribution by Class')
        ax.legend()
        self._finish(fig, 'score_distribution')

    def ks_plot(self):
        r, f = np.sort(self.p[self.y == 0]), np.sort(self.p[self.y == 1])
        grid = np.linspace(0, 1, 1001)
        cr = np.searchsorted(r, grid, side='right') / len(r)
        cf = np.searchsorted(f, grid, side='right') / len(f)
        k = int(np.argmax(np.abs(cr - cf)))
        fig, ax = plt.subplots(figsize=SQ)
        ax.plot(grid, cr, color=C['green'], label='Real'); ax.plot(grid, cf, color=C['red'], label='Fake')
        ax.vlines(grid[k], cf[k], cr[k], color='k', lw=2.5, label=f"KS = {self.m['KS Statistic']:.4f} @ {grid[k]:.3f}")
        self._style(ax, 'Predicted probability of Fake', 'Empirical CDF', 'Kolmogorov-Smirnov Separation')
        ax.legend(loc='center right')
        self._finish(fig, 'ks_statistic')

    def threshold_analysis(self):
        ths = np.linspace(0, 1, 201)
        keys = [('Accuracy', C['blue']), ('Precision', C['green']), ('Recall', C['red']), ('F1-Score', C['purple']),
                ('MCC', C['orange']), ("Cohen's Kappa", C['sky'])]
        rows = [quick_metrics(*_confusion(self.y, (self.p >= t).astype(int))) for t in ths]
        fig, ax = plt.subplots(figsize=WIDE)
        for k, c in keys:
            ax.plot(ths, [r[k] for r in rows], color=c, label=k)
        bt = ths[int(np.argmax([r['F1-Score'] for r in rows]))]
        ax.axvline(bt, color='k', ls='--', label=f'Best-F1 threshold = {bt:.3f}')
        ax.axvline(0.5, color=C['grey'], ls=':', label='Default = 0.5')
        self._style(ax, 'Decision threshold', 'Score', 'Metrics vs. Decision Threshold')
        ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.03)
        ax.legend(ncol=2, loc='lower center')
        self._finish(fig, 'threshold_analysis')

    def metrics_bar(self):
        keys = ['Accuracy', 'Balanced Acc.', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC-ROC', 'MCC', "Cohen's Kappa"]
        vals = [self.m[k] for k in keys]
        cols = [C['blue'], C['sky'], C['green'], C['red'], C['orange'], C['purple'], C['black'], C['grey'], '#8C564B']
        err = np.array([[self.m[k] - self.R['ci'][k][0], self.R['ci'][k][1] - self.m[k]] if k in self.R['ci'] else [0, 0]
                        for k in keys]).T
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        bars = ax.bar(keys, vals, color=cols, edgecolor='k', lw=1.2, alpha=0.9, yerr=np.clip(err, 0, None), capsize=5,
                      error_kw={'lw': 1.6})
        self._bar_labels(ax, bars, '{:.4f}', 10, 0.015)
        ax.axhline(0.9, ls='--', color=C['red'], alpha=0.7, label='90% target')
        self._style(ax, None, 'Score', 'Classification Performance (error bars: 95% bootstrap CI)')
        ax.set_ylim(0, 1.12); plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        ax.legend(loc='lower right')
        self._finish(fig, 'classification_metrics_bar')

    def per_class(self):
        rep = classification_report(self.y, self.pred, target_names=CLASSES, output_dict=True, zero_division=0)
        mets = [('precision', 'Precision', C['blue']), ('recall', 'Recall', C['red']), ('f1-score', 'F1-Score', C['green'])]
        x, w = np.arange(2), 0.26
        fig, ax = plt.subplots(figsize=SQ)
        for i, (k, nm, c) in enumerate(mets):
            bars = ax.bar(x + (i - 1) * w, [rep[cl][k] for cl in CLASSES], w, label=nm, color=c, edgecolor='k', alpha=0.9)
            self._bar_labels(ax, bars, '{:.3f}', 10)
        ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={int(rep[c]['support'])})" for c in CLASSES])
        self._style(ax, None, 'Score', 'Per-class Performance')
        ax.set_ylim(0, 1.15); ax.legend(loc='lower right')
        self._finish(fig, 'per_class_metrics')

    def radar(self):
        labs = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC-ROC']
        ang = np.linspace(0, 2 * np.pi, len(labs), endpoint=False).tolist(); ang += ang[:1]
        fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(111, polar=True)
        allv = []
        for n in MODELS:
            v = [self.R['metrics'][n][l] for l in labs]; allv += v
            ax.plot(ang, v + v[:1], color=MODEL_COLORS[n], label=n, lw=3 if n == 'Ensemble' else 2)
            ax.fill(ang, v + v[:1], color=MODEL_COLORS[n], alpha=0.07)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(labs, fontsize=12, fontweight='bold')
        ax.set_ylim(max(0, math.floor(min(allv) * 20) / 20 - 0.05), 1.0)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
        if Config.SHOW_TITLES: ax.set_title('Radar Comparison of Models', fontweight='bold', pad=25)
        self._finish(fig, 'radar_models')

    def model_comparison(self):
        keys = ['Accuracy', 'F1-Score', 'AUC-ROC', 'MCC', "Cohen's Kappa"]
        x, w = np.arange(len(keys)), 0.2
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        for i, n in enumerate(MODELS):
            bars = ax.bar(x + (i - 1.5) * w, [self.R['metrics'][n][k] for k in keys], w, label=n,
                          color=MODEL_COLORS[n], edgecolor='k', alpha=0.9)
            self._bar_labels(ax, bars, '{:.3f}', 8.5, 0.008)
        lo = min(self.R['metrics'][n][k] for n in MODELS for k in keys)
        ax.set_xticks(x); ax.set_xticklabels(keys)
        ax.set_ylim(max(0, lo - 0.1), 1.07)
        self._style(ax, None, 'Score', 'Individual Models vs. Ensemble')
        ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))
        self._finish(fig, 'model_comparison')

    def bootstrap_box(self):
        keys = ['Accuracy', 'F1-Score', 'AUC-ROC', 'MCC', "Cohen's Kappa"]
        data = [self.R['boot'][k] for k in keys]
        fig, ax = plt.subplots(figsize=WIDE)
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showmeans=True,
                        meanprops=dict(marker='D', mfc='white', mec='k', ms=7), medianprops=dict(color='k', lw=2))
        for patch, c in zip(bp['boxes'], [C['blue'], C['purple'], C['black'], C['grey'], '#8C564B']):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_xticklabels(keys)
        self._style(ax, None, 'Score', f'Bootstrap Distributions ({Config.BOOTSTRAP_ITERS} resamples)')
        self._finish(fig, 'bootstrap_distributions')

    # ---------- training-history figures ----------
    def _hx(self, key):
        v = np.asarray(self.R['history'].get(key, []), dtype=float)
        return np.arange(1, len(v) + 1), v

    def train_loss(self):
        fig, ax = plt.subplots(figsize=WIDE)
        for k, nm, c in [('train_loss', 'Train', C['blue']), ('val_loss', 'Validation', C['red'])]:
            x, v = self._hx(k); ax.plot(x, v, color=c, label=nm)
        self._style(ax, 'Epoch', 'Loss', 'Training and Validation Loss'); ax.legend()
        self._finish(fig, 'training_loss')

    def train_acc(self):
        fig, ax = plt.subplots(figsize=WIDE)
        for k, nm, c in [('train_acc', 'Train', C['blue']), ('val_acc', 'Validation', C['red'])]:
            x, v = self._hx(k); ax.plot(x, v, color=c, label=nm)
        x, v = self._hx('val_acc')
        if len(v):
            b = int(np.nanargmax(v))
            ax.plot(x[b], v[b], '*', ms=20, color=C['orange'], mec='k', label=f'Best val = {v[b]:.4f} (epoch {x[b]})')
        ax.axhline(0.9, ls='--', color=C['green'], label='90% target')
        self._style(ax, 'Epoch', 'Accuracy', 'Training and Validation Accuracy'); ax.legend(loc='lower right')
        self._finish(fig, 'training_accuracy')

    def val_metrics_curve(self):
        fig, ax = plt.subplots(figsize=WIDE)
        for k, nm, c in [('val_f1', 'F1-Score', C['purple']), ('val_auc', 'AUC-ROC', C['black']),
                         ('val_mcc', 'MCC', C['orange']), ('val_kappa', "Cohen's Kappa", C['sky'])]:
            x, v = self._hx(k)
            if len(v): ax.plot(x, v, color=c, label=nm)
        self._style(ax, 'Epoch', 'Validation score', 'Validation Metrics per Epoch'); ax.legend(loc='lower right')
        self._finish(fig, 'validation_metrics_per_epoch')

    def lr_curve(self):
        x, v = self._hx('lr')
        if not len(v) or np.all(np.isnan(v)): return
        fig, ax = plt.subplots(figsize=WIDE)
        ax.semilogy(x, v, color=C['blue'], marker='o', ms=4)
        self._style(ax, 'Epoch', 'Learning rate (new layers)', 'Learning-rate Schedule (cosine w/ warm restarts)')
        self._finish(fig, 'learning_rate_schedule')

    def gen_gap(self):
        x, t = self._hx('train_acc'); _, v = self._hx('val_acc')
        fig, ax = plt.subplots(figsize=WIDE)
        ax.plot(x, t - v, color=C['red'])
        ax.fill_between(x, 0, t - v, color=C['red'], alpha=0.2); ax.axhline(0, color='k', lw=1)
        self._style(ax, 'Epoch', 'Train acc. - Val acc.', 'Generalisation Gap')
        self._finish(fig, 'generalization_gap')

    def epoch_time(self):
        x, tt = self._hx('train_time'); _, vt = self._hx('val_time')
        if not len(tt) or np.all(np.isnan(tt)): return
        tt, vt = np.nan_to_num(tt), np.nan_to_num(vt)
        fig, ax = plt.subplots(figsize=WIDE)
        ax.bar(x, tt, color=C['blue'], edgecolor='k', lw=0.4, label='Training')
        ax.bar(x, vt, bottom=tt, color=C['orange'], edgecolor='k', lw=0.4, label='Validation')
        self._style(ax, 'Epoch', 'Wall-clock time per epoch (s)', 'Epoch Time Breakdown'); ax.legend()
        self._finish(fig, 'epoch_time_breakdown')

    def train_throughput(self):
        x, v = self._hx('train_throughput')
        if not len(v) or np.all(np.isnan(v)): return
        fig, ax = plt.subplots(figsize=WIDE)
        ax.plot(x, v, color=C['green'], marker='o', ms=4)
        ax.axhline(np.nanmean(v), ls='--', color='k', label=f'Mean = {np.nanmean(v):.2f} videos/s'); ax.legend()
        self._style(ax, 'Epoch', 'Training throughput (videos/s)', 'Training Throughput')
        self._finish(fig, 'training_throughput')

    # ---------- efficiency figures ----------
    def time_summary(self):
        T = self.R['timing']
        items = [('Total training', T['training_total_s']), ('Mean epoch\n(train)', T['train_epoch_mean_s']),
                 ('Mean epoch\n(validation)', T['val_epoch_mean_s']), (f"Testing\n(TTA x{T['tta_iters']})", T['test_wall_s']),
                 ('Test model\ncompute', T['test_compute_s'])]
        items = [(k, v) for k, v in items if v and np.isfinite(v)]
        fig, ax = plt.subplots(figsize=WIDE)
        bars = ax.bar([k for k, _ in items], [v for _, v in items],
                      color=[C['blue'], C['sky'], C['orange'], C['red'], C['purple']][:len(items)], edgecolor='k')
        ax.set_yscale('log')
        for b, (_, v) in zip(bars, items):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, fmt_time(v), ha='center', fontweight='bold')
        ax.set_ylim(top=max(v for _, v in items) * 6)
        self._style(ax, None, 'Time (seconds, log scale)', 'Training / Validation / Testing Time')
        self._finish(fig, 'time_summary')

    def latency_hist(self):
        t = self.R['bench']['single_ms']['Ensemble']
        fig, ax = plt.subplots(figsize=WIDE)
        ax.hist(t, bins=30, color=C['blue'], alpha=0.75, edgecolor='k')
        for q, c, ls in [(np.mean(t), C['red'], '-'), (np.percentile(t, 95), C['orange'], '--'), (np.percentile(t, 99), C['purple'], ':')]:
            ax.axvline(q, color=c, ls=ls, lw=2.5)
        ax.legend([f'Mean = {t.mean():.2f} ms', f'P95 = {np.percentile(t, 95):.2f} ms', f'P99 = {np.percentile(t, 99):.2f} ms'],
                  title=f'std = {t.std():.2f} ms, n = {len(t)}')
        self._style(ax, 'Single-video inference latency (ms, batch = 1)', 'Count', 'Inference Latency Distribution (Ensemble)')
        self._finish(fig, 'latency_distribution')

    def throughput_curve(self):
        sw = self.R['bench']['sweep']; bs = sorted(sw)
        fig, ax = plt.subplots(figsize=SQ)
        ax.plot(bs, [sw[b]['throughput_vps'] for b in bs], 'o-', color=C['green'], ms=9)
        for b in bs:
            ax.annotate(f"{sw[b]['throughput_vps']:.1f}", (b, sw[b]['throughput_vps']), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontweight='bold')
        ax.set_xscale('log', base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
        self._style(ax, 'Batch size', 'Throughput (videos / s)', 'Inference Throughput vs. Batch Size')
        ax.margins(y=0.15)
        self._finish(fig, 'throughput_vs_batch')

    def latency_curve(self):
        sw = self.R['bench']['sweep']; bs = sorted(sw)
        fig, ax = plt.subplots(figsize=SQ)
        ax.errorbar(bs, [sw[b]['ms_per_video'] for b in bs], yerr=[sw[b]['std_ms_per_video'] for b in bs],
                    fmt='s-', color=C['red'], ms=8, capsize=4)
        ax.set_xscale('log', base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
        self._style(ax, 'Batch size', 'Latency per video (ms)', 'Per-video Latency vs. Batch Size')
        self._finish(fig, 'latency_vs_batch')

    def component_latency(self):
        s = self.R['bench']['single_ms']
        fig, ax = plt.subplots(figsize=SQ)
        bars = ax.bar(MODELS, [s[n].mean() for n in MODELS], yerr=[s[n].std() for n in MODELS], capsize=6,
                      color=[MODEL_COLORS[n] for n in MODELS], edgecolor='k', alpha=0.9)
        self._bar_labels(ax, bars, '{:.2f}', 11, max(s[n].mean() for n in MODELS) * 0.02)
        self._style(ax, None, 'Latency per video (ms, batch = 1)', 'Per-component Inference Latency')
        self._finish(fig, 'component_latency')

    def params_bar(self):
        cx = self.R['bench']['complexity']; x, w = np.arange(4), 0.38
        fig, ax = plt.subplots(figsize=SQ)
        b1 = ax.bar(x - w / 2, [cx[n]['params_M'] for n in MODELS], w, label='Total', color=C['blue'], edgecolor='k')
        b2 = ax.bar(x + w / 2, [cx[n]['trainable_M'] for n in MODELS], w, label='Trainable', color=C['orange'], edgecolor='k')
        for bars in (b1, b2): self._bar_labels(ax, bars, '{:.1f}', 10, 0.5)
        ax.set_xticks(x); ax.set_xticklabels(MODELS)
        self._style(ax, None, 'Parameters (millions)', 'Model Size'); ax.legend()
        self._finish(fig, 'parameters')

    def flops_bar(self):
        cx = self.R['bench']['complexity']
        v = [cx[n]['GFLOPs'] for n in MODELS]
        if not np.all(np.isfinite(v)): return
        fig, ax = plt.subplots(figsize=SQ)
        bars = ax.bar(MODELS, v, color=[MODEL_COLORS[n] for n in MODELS], edgecolor='k')
        self._bar_labels(ax, bars, '{:.1f}', 11, max(v) * 0.01)
        self._style(ax, None, 'GFLOPs per video (batch = 1)', 'Computational Cost (LSTM excluded)')
        self._finish(fig, 'gflops')

    def acc_vs_latency(self):
        s, cx = self.R['bench']['single_ms'], self.R['bench']['complexity']
        fig, ax = plt.subplots(figsize=SQ)
        for n in MODELS:
            ax.scatter(s[n].mean(), self.R['metrics'][n]['Accuracy'], s=60 + 12 * cx[n]['params_M'],
                       color=MODEL_COLORS[n], edgecolor='k', alpha=0.85, label=f"{n} ({cx[n]['params_M']:.1f} M)", zorder=3)
            ax.annotate(n, (s[n].mean(), self.R['metrics'][n]['Accuracy']), textcoords='offset points',
                        xytext=(8, 8), fontweight='bold')
        self._style(ax, 'Latency per video (ms)', 'Accuracy', 'Accuracy vs. Latency (marker area ~ #params)')
        ax.margins(0.2); ax.legend(loc='lower right', title='Model (params)')
        self._finish(fig, 'accuracy_vs_latency')

    # ---------- table figures ----------
    def metrics_table(self):
        rows = ['Accuracy', 'Balanced Acc.', 'Precision', 'Recall', 'Specificity', 'NPV', 'F1-Score', 'MCC',
                "Cohen's Kappa", 'G-Mean', 'AUC-ROC', 'AP (AUC-PR)', 'EER', 'Log Loss', 'Brier Score', 'ECE']
        df = pd.DataFrame({n: [f"{self.R['metrics'][n].get(r, np.nan):.4f}" for r in rows] for n in MODELS}, index=rows)
        df['Ensemble 95% CI'] = [f"[{self.R['ci'][r][0]:.3f}, {self.R['ci'][r][1]:.3f}]" if r in self.R['ci'] else '-' for r in rows]
        self._table_fig(df, 'table_performance_metrics', 'Test-set Performance Summary', 1.7)

    def efficiency_table(self):
        cx, s, sw = self.R['bench']['complexity'], self.R['bench']['single_ms'], self.R['bench']['sweep']
        b1 = sw.get(1, {}).get('throughput_vps', float('nan'))
        rows = {
            'Parameters (M)': [f"{cx[n]['params_M']:.2f}" for n in MODELS],
            'Trainable (M)': [f"{cx[n]['trainable_M']:.2f}" for n in MODELS],
            'Model size (MB)': [f"{cx[n]['size_MB']:.1f}" for n in MODELS],
            'GFLOPs / video': [f"{cx[n]['GFLOPs']:.1f}" for n in MODELS],
            'Latency mean (ms)': [f"{s[n].mean():.2f}" for n in MODELS],
            'Latency P95 (ms)': [f"{np.percentile(s[n], 95):.2f}" for n in MODELS],
            'Throughput bs=1 (vid/s)': [f"{1000 / s[n].mean():.2f}" for n in MODELS],
        }
        self._table_fig(pd.DataFrame(rows, index=MODELS).T, 'table_efficiency', 'Efficiency Summary', 1.6)

    # ---------- run everything ----------
    def plot_all(self):
        print("\nGenerating figures (display -> save @ %d dpi)..." % Config.DPI)
        for fn in [self.confusion_counts, self.confusion_normalized, self.roc, self.roc_semilog, self.det, self.pr,
                   self.calibration, self.score_distribution, self.ks_plot, self.threshold_analysis, self.metrics_bar,
                   self.per_class, self.radar, self.model_comparison, self.bootstrap_box, self.train_loss,
                   self.train_acc, self.val_metrics_curve, self.lr_curve, self.gen_gap, self.epoch_time,
                   self.train_throughput, self.time_summary, self.latency_hist, self.throughput_curve,
                   self.latency_curve, self.component_latency, self.params_bar, self.flops_bar,
                   self.acc_vs_latency, self.metrics_table, self.efficiency_table]:
            try:
                fn()
            except Exception as e:
                plt.close('all')
                print(f"  ! {fn.__name__} skipped: {type(e).__name__}: {e}")
        print(f"\n✓ {self.n} figures saved to {self.dir}")
# <<< PLOTTER_END

# =====================================================================
# >>> PLOTTER2_BEGIN
# Figures for reproducibility / thresholds / weights / significance / robustness
# (same pipeline: display -> save 600 dpi PNG + vector PDF)
# =====================================================================
SEED_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#7F7F7F', '#F0E442']
THR_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
MAIN_KEYS = ['Accuracy', 'F1-Score', 'AUC-ROC', 'MCC', "Cohen's Kappa"]

def _pad_stack(list_of_lists):
    L = max(len(v) for v in list_of_lists)
    A = np.full((len(list_of_lists), L), np.nan)
    for i, v in enumerate(list_of_lists):
        A[i, :len(v)] = np.asarray(v, float)
    return A

class AnalysisPlotter(_FigBase):
    def __init__(self, save_dir, per_seed=None, primary_seed=None):
        super().__init__(save_dir, '')
        self.per_seed, self.primary = per_seed or {}, primary_seed

    def group(self, prefix):
        self.prefix, self.n = prefix, 0
        return self

    # ================= reproducibility =================
    def seed_bars(self, df_seed, model='Ensemble'):
        d = df_seed[df_seed.model == model]
        seeds = list(d.seed)
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        x = np.arange(len(MAIN_KEYS)); w = 0.8 / (len(seeds) + 1)
        for i, s in enumerate(seeds):
            vals = [d[d.seed == s][k].iloc[0] for k in MAIN_KEYS]
            b = ax.bar(x + (i - len(seeds) / 2) * w + w / 2, vals, w, color=SEED_COLORS[i % 8], edgecolor='k', label=f'seed {s}')
            self._bar_labels(ax, b, '{:.3f}', 7.5, 0.004)
        mu = [d[k].mean() for k in MAIN_KEYS]; sd = [d[k].std(ddof=1) if len(d) > 1 else 0 for k in MAIN_KEYS]
        b = ax.bar(x + (len(seeds) / 2) * w + w / 2, mu, w, yerr=sd, capsize=4, color='white', edgecolor='k', hatch='//', label='mean ± std')
        ax.set_xticks(x); ax.set_xticklabels(MAIN_KEYS)
        lo = min(d[k].min() for k in MAIN_KEYS)
        ax.set_ylim(max(0, lo - 0.08), 1.03)
        self._style(ax, None, 'Score', f'Reproducibility across seeds ({model})')
        ax.legend(ncol=len(seeds) + 1, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=9)
        self._finish(fig, 'seed_metrics_bars')

    def curves_over_seeds(self, key, ylabel, slug, title, best_max=True):
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        arrs = []
        for i, (s, S) in enumerate(self.per_seed.items()):
            v = np.asarray(S['history'][key], float)
            ax.plot(np.arange(1, len(v) + 1), v, color=SEED_COLORS[i % 8], alpha=0.75, lw=1.6, label=f'seed {s}')
            arrs.append(v)
        A = _pad_stack(arrs)
        if A.shape[0] > 1:
            mu, sd = np.nanmean(A, 0), np.nanstd(A, 0, ddof=1)
            x = np.arange(1, A.shape[1] + 1)
            ax.plot(x, mu, 'k-', lw=2.8, label='mean'); ax.fill_between(x, mu - sd, mu + sd, color='k', alpha=0.15, label='± std')
        self._style(ax, 'Epoch', ylabel, title); ax.legend()
        self._finish(fig, slug)

    def roc_over_seeds(self):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        grid = np.linspace(0, 1, 501); tprs, aucs = [], []
        for i, (s, S) in enumerate(self.per_seed.items()):
            f, t, _ = roc_curve(S['y_true'], S['probs']['Ensemble'])
            a = roc_auc_score(S['y_true'], S['probs']['Ensemble']); aucs.append(a)
            ax.plot(f, t, color=SEED_COLORS[i % 8], alpha=0.7, lw=1.6, label=f'seed {s} (AUC={a:.4f})')
            tprs.append(np.interp(grid, f, t))
        mt = np.mean(tprs, 0); st = np.std(tprs, 0, ddof=1) if len(tprs) > 1 else np.zeros_like(mt)
        ax.plot(grid, mt, 'k-', lw=2.6, label=f'mean (AUC={np.mean(aucs):.4f} ± {np.std(aucs, ddof=1) if len(aucs) > 1 else 0:.4f})')
        ax.fill_between(grid, np.clip(mt - st, 0, 1), np.clip(mt + st, 0, 1), color='k', alpha=0.15)
        ax.plot([0, 1], [0, 1], '--', color=C['grey'])
        self._style(ax, 'False Positive Rate', 'True Positive Rate', 'ROC across seeds'); ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
        self._finish(fig, 'roc_across_seeds')

    def repro_table(self, summ):
        keys = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC', "Cohen's Kappa", 'AUC-ROC']
        df = pd.DataFrame({m: [f"{summ[(summ.model == m) & (summ.metric == k)]['mean'].iloc[0]:.4f} ± "
                               f"{summ[(summ.model == m) & (summ.metric == k)]['std'].iloc[0]:.4f}" for k in keys]
                           for m in MODELS}, index=keys)
        df['Ensemble 95% CI'] = [f"[{summ[(summ.model == 'Ensemble') & (summ.metric == k)]['ci95_low'].iloc[0]:.4f}, "
                                 f"{summ[(summ.model == 'Ensemble') & (summ.metric == k)]['ci95_high'].iloc[0]:.4f}]" for k in keys]
        self._table_fig(df, 'table_reproducibility', f'Mean ± std over {int(summ.n_seeds.max())} seeds', 2.0)

    def repeat_table(self, rep_df):
        if rep_df is None or not len(rep_df):
            return
        df = rep_df.copy()
        for c in df.columns:
            if df[c].dtype.kind == 'f':
                df[c] = df[c].map(lambda v: f"{v:.3g}")
        df = df.set_index(df.columns[0]).astype(str)
        self._table_fig(df, 'table_repeatability', 'Repeatability (same seed, repeated inference)', 2.0)

    # ================= thresholds =================
    def threshold_bars(self, thr_df):
        keys = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC', "Cohen's Kappa"]
        strategies = list(dict.fromkeys(thr_df.strategy))
        fig, ax = plt.subplots(figsize=(11, 5.8))
        x = np.arange(len(keys)); w = 0.8 / len(strategies)
        for i, s in enumerate(strategies):
            sub = thr_df[thr_df.strategy == s]
            mu = [sub[k].mean() for k in keys]; sd = [sub[k].std(ddof=1) if len(sub) > 1 else 0 for k in keys]
            b = ax.bar(x + (i - len(strategies) / 2) * w + w / 2, mu, w, yerr=sd, capsize=3, color=THR_COLORS[i % 5], edgecolor='k',
                       label=f"{THRESHOLD_LABELS.get(s, s)}  (τ̄={sub.threshold.mean():.3f})")
            self._bar_labels(ax, b, '{:.3f}', 7, 0.006)
        ax.set_xticks(x); ax.set_xticklabels(keys)
        lo = min(thr_df[k].min() for k in keys)
        ax.set_ylim(max(0, lo - 0.1), 1.08)
        self._style(ax, None, 'Score (mean ± std over seeds)', 'Ensemble performance at four decision thresholds')
        ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=9)
        self._finish(fig, 'threshold_metrics_bars')

    def threshold_confusions(self, y, p, thr_map):
        n = len(thr_map)
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4))
        for ax, (name, t) in zip(np.atleast_1d(axes), thr_map.items()):
            cm = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, ax=ax, linewidths=1.2, linecolor='white',
                        xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={'size': 17, 'weight': 'bold'})
            ax.set_xlabel('Predicted', fontweight='bold'); ax.set_ylabel('True', fontweight='bold')
            if Config.SHOW_TITLES: ax.set_title(f"{THRESHOLD_LABELS.get(name, name)}\nτ = {t:.3f}", fontweight='bold', fontsize=12)
        self._finish(fig, 'threshold_confusion_matrices')

    def threshold_roc(self, y, p, thr_map):
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.plot(fpr, tpr, color=C['grey'], lw=2.4, label=f'ROC (AUC = {roc_auc_score(y, p):.4f})')
        mk = ['o', 's', '^', 'D', 'v']
        for i, (name, t) in enumerate(thr_map.items()):
            pr = (p >= t).astype(int); tp, tn, fp, fn = _confusion(y, pr)
            ax.plot(fp / max(fp + tn, 1), tp / max(tp + fn, 1), mk[i % 5], ms=11, color=THR_COLORS[i % 5], mec='k',
                    label=f"{THRESHOLD_LABELS.get(name, name)}: τ={t:.3f}")
        ax.plot([0, 1], [0, 1], ':', color='k', lw=1)
        self._style(ax, 'False Positive Rate', 'True Positive Rate', 'Operating points on the ROC curve')
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02); ax.legend(loc='lower right', fontsize=9)
        self._finish(fig, 'threshold_operating_points')

    def threshold_stability(self, thr_df):
        strategies = list(dict.fromkeys(thr_df.strategy))
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for i, s in enumerate(strategies):
            sub = thr_df[thr_df.strategy == s]
            ax.scatter([i] * len(sub), sub.threshold, s=70, color=THR_COLORS[i % 5], edgecolor='k', zorder=3)
            ax.errorbar(i, sub.threshold.mean(), yerr=sub.threshold.std(ddof=1) if len(sub) > 1 else 0, fmt='_', color='k', ms=26, capsize=6)
        ax.set_xticks(range(len(strategies))); ax.set_xticklabels([THRESHOLD_LABELS.get(s, s) for s in strategies], rotation=12)
        self._style(ax, None, 'Selected threshold τ', 'Stability of the threshold across seeds (dots = seeds)')
        self._finish(fig, 'threshold_stability')

    def threshold_table(self, thr_df):
        keys = ['threshold', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC', "Cohen's Kappa", 'FPR', 'FNR']
        rows = {}
        for s in dict.fromkeys(thr_df.strategy):
            sub = thr_df[thr_df.strategy == s]
            rows[THRESHOLD_LABELS.get(s, s)] = [f"{sub[k].mean():.4f} ± {sub[k].std(ddof=1) if len(sub) > 1 else 0:.4f}" for k in keys]
        self._table_fig(pd.DataFrame(rows, index=keys).T, 'table_thresholds', 'Threshold comparison (mean ± std over seeds)', 1.8)

    # ================= ensemble weights =================
    def weight_bars(self, w_df):
        keys = ['Accuracy', 'F1-Score', 'AUC-ROC', 'MCC']
        cfgs = list(dict.fromkeys(w_df.config))
        fig, ax = plt.subplots(figsize=(12.5, 5.8))
        x = np.arange(len(cfgs)); w = 0.8 / len(keys)
        for i, k in enumerate(keys):
            mu = [w_df[w_df.config == c][k].mean() for c in cfgs]
            sd = [w_df[w_df.config == c][k].std(ddof=1) if (w_df.config == c).sum() > 1 else 0 for c in cfgs]
            ax.bar(x + (i - len(keys) / 2) * w + w / 2, mu, w, yerr=sd, capsize=2.5, label=k, color=THR_COLORS[i], edgecolor='k')
        ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=25, ha='right')
        lo = min(w_df[k].min() for k in keys)
        ax.set_ylim(max(0, lo - 0.08), 1.04)
        self._style(ax, None, 'Score (mean ± std over seeds)', 'Ensemble weighting schemes (test set)')
        ax.legend(ncol=4, loc='lower right', fontsize=9)
        self._finish(fig, 'weight_comparison_bars')

    def weight_simplex(self, gdf, w_df_seed):
        V = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
        fig, ax = plt.subplots(figsize=(7.4, 6.4))
        Wm = gdf[['w_video', 'w_audio', 'w_fusion']].to_numpy()
        xy = Wm @ V
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=gdf.test_acc, cmap='viridis', s=120, marker='h', edgecolor='none')
        plt.colorbar(sc, ax=ax, label='Test accuracy', shrink=0.8)
        ax.plot(*np.vstack([V, V[0]]).T, 'k-', lw=1.5)
        for (px, py), lab, dx, dy in zip(V, ['Video = 1', 'Audio = 1', 'Fusion = 1'], [-0.02, 0.02, 0], [-0.05, -0.05, 0.04]):
            ax.text(px + dx, py + dy, lab, ha='center', fontweight='bold')
        mk = {'Default': ('*', C['red'], 18), 'Val-optimised': ('P', C['orange'], 13), 'Equal': ('X', 'white', 12)}
        for name, (m, c, s) in mk.items():
            r = w_df_seed[w_df_seed.config == name]
            if len(r):
                pt = r[['w_video', 'w_audio', 'w_fusion']].to_numpy()[0] @ V
                ax.plot(*pt, m, color=c, ms=s, mec='k', label=name)
        ax.set_aspect('equal'); ax.axis('off'); ax.legend(loc='upper left')
        if Config.SHOW_TITLES: ax.set_title('Test accuracy over the weight simplex (primary seed)', fontweight='bold')
        self._finish(fig, 'weight_simplex')

    def weight_roc(self, y, probs):
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        for i, (name, p) in enumerate(probs.items()):
            f, t, _ = roc_curve(y, p)
            ax.plot(f, t, lw=2.8 if name == 'Default' else 1.6, color=SEED_COLORS[i % 8] if i < 8 else None,
                    ls='-' if i < 8 else '--', label=f"{name} ({roc_auc_score(y, p):.4f})")
        ax.plot([0, 1], [0, 1], ':', color='k')
        ax.set_xscale('symlog', linthresh=0.01); ax.set_xlim(0, 1)
        self._style(ax, 'False Positive Rate (symlog)', 'True Positive Rate', 'ROC of weighting schemes (primary seed)')
        ax.legend(fontsize=8, loc='lower right', ncol=2)
        self._finish(fig, 'weight_roc')

    def weight_table(self, w_df):
        keys = ['w_video', 'w_audio', 'w_fusion', 'Accuracy', 'F1-Score', 'AUC-ROC', 'MCC']
        rows = {}
        for c in dict.fromkeys(w_df.config):
            sub = w_df[w_df.config == c]
            rows[c] = [f"{sub[k].mean():.3f}" if k.startswith('w_') and sub[k].notna().all() else
                       ('-' if k.startswith('w_') else f"{sub[k].mean():.4f} ± {sub[k].std(ddof=1) if len(sub) > 1 else 0:.4f}") for k in keys]
        self._table_fig(pd.DataFrame(rows, index=keys).T, 'table_weights', 'Ensemble-weight comparison', 1.6)

    # ================= significance =================
    def forest_across_seeds(self, sig_df):
        d = sig_df[sig_df.metric.isin(['Accuracy', 'F1-Score', 'AUC-ROC', 'MCC'])].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(d) + 2.2))
        y = np.arange(len(d))[::-1]
        cols = {'Accuracy': C['blue'], 'F1-Score': C['green'], 'AUC-ROC': C['red'], 'MCC': C['orange']}
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.errorbar(r.mean_diff, yi, xerr=[[r.mean_diff - r.diff_ci95_low], [r.diff_ci95_high - r.mean_diff]], fmt='o',
                        color=cols[r.metric], capsize=4, ms=7)
            ax.text(ax.get_xlim()[1], yi, f"  {r.t_sig_holm}", va='center', fontweight='bold')
        ax.axvline(0, color='k', lw=1)
        ax.set_yticks(y); ax.set_yticklabels([f"{r.metric}: {r.A} − {r.B}" for _, r in d.iterrows()], fontsize=9)
        self._style(ax, 'Mean paired difference over seeds (95% CI)', None, 'Seed-level significance (Holm-adjusted paired t-test)')
        ax.margins(x=0.15)
        self._finish(fig, 'forest_across_seeds')

    def forest_per_seed(self, sig_seed_df):
        d = sig_seed_df.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.4, 0.38 * len(d) + 2.2))
        y = np.arange(len(d))[::-1]
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.errorbar(r.acc_diff, yi, xerr=[[r.acc_diff - r.acc_diff_ci_low], [r.acc_diff_ci_high - r.acc_diff]], fmt='s', color=C['blue'], capsize=4)
            ax.text(ax.get_xlim()[1], yi, f"  McN {r.mcnemar_sig}", va='center', fontsize=9)
        ax.axvline(0, color='k', lw=1)
        ax.set_yticks(y); ax.set_yticklabels([f"seed {r.seed}: {r.A} − {r.B}" for _, r in d.iterrows()], fontsize=9)
        self._style(ax, 'Accuracy difference on the test set (95% paired-bootstrap CI)', None, 'Per-seed paired comparison')
        ax.margins(x=0.2)
        self._finish(fig, 'forest_per_seed')

    def pairwise_pvalues(self, names, M, seed):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        L = -np.log10(np.clip(M, 1e-12, 1))
        np.fill_diagonal(L, np.nan)
        annot = np.array([[('-' if i == j else (f"{M[i, j]:.1e}" if M[i, j] < 1e-3 else f"{M[i, j]:.3f}") + sig_stars(M[i, j]).replace('ns', ''))
                           for j in range(len(names))] for i in range(len(names))])
        sns.heatmap(L, annot=annot, fmt='', cmap='Reds', ax=ax, square=True, xticklabels=names, yticklabels=names,
                    cbar_kws={'label': '−log10(p)'}, linewidths=1, linecolor='white', vmin=0, vmax=max(2.0, float(np.nanmax(L))))
        self._style(ax, None, None, f'Pairwise McNemar tests (seed {seed})', grid=False)
        self._finish(fig, 'pairwise_mcnemar')

    def seed_boxplots(self, df_seed, sig_df):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        for ax, k in zip(axes, ['Accuracy', 'AUC-ROC']):
            data = [df_seed[df_seed.model == m][k].to_numpy(float) for m in MODELS]
            bp = ax.boxplot(data, patch_artist=True, widths=0.5, showmeans=True,
                            meanprops=dict(marker='D', mfc='white', mec='k', ms=7), medianprops=dict(color='k'))
            for patch, m in zip(bp['boxes'], MODELS):
                patch.set_facecolor(MODEL_COLORS[m]); patch.set_alpha(0.55)
            for i, d_ in enumerate(data, start=1):
                ax.scatter(np.full(len(d_), i) + np.linspace(-0.08, 0.08, len(d_)), d_, color='k', s=22, zorder=3)
            ax.set_xticklabels(MODELS)
            top = max(np.max(d_) for d_ in data); bot = min(np.min(d_) for d_ in data)
            span = max(top - bot, 0.02)
            for j, m in enumerate(MODELS[:-1], start=1):
                r = sig_df[(sig_df.metric == k) & (sig_df.B == m)]
                if len(r):
                    yy = top + span * (0.12 + 0.12 * (j - 1))
                    ax.plot([j, j, 4, 4], [yy - span * .03, yy, yy, yy - span * .03], 'k-', lw=1)
                    ax.text((j + 4) / 2, yy, r.t_sig_holm.iloc[0], ha='center', va='bottom', fontweight='bold')
            ax.set_ylim(bot - span * 0.2, top + span * 0.75)
            self._style(ax, None, k, f'{k} over seeds (Ensemble vs. others)')
        self._finish(fig, 'seed_boxplots_significance')

    def sig_table(self, sig_df):
        d = sig_df.copy()
        rows = {f"{r.metric}: {r.A} vs {r.B}": [f"{r.mean_diff:+.4f}", f"[{r.diff_ci95_low:+.4f}, {r.diff_ci95_high:+.4f}]",
                                                f"{r.cohens_dz:.2f}", f"{r.paired_t_p:.4f}", f"{r.paired_t_p_holm:.4f} {r.t_sig_holm}", f"{r.wilcoxon_p:.3f}"]
                for _, r in d.iterrows()}
        self._table_fig(pd.DataFrame(rows, index=['Mean diff', '95% CI', "Cohen's dz", 'p (t)', 'p (Holm)', 'p (Wilcoxon)']).T,
                        'table_significance', 'Ensemble vs. components across seeds', 1.7)

    # ================= robustness =================
    def _rob_mean(self, rob_df, model='Ensemble', col='Accuracy'):
        d = rob_df[rob_df.model == model]
        return d.groupby(['perturbation', 'level'], sort=False)[col].agg(['mean', 'std']).fillna(0)

    def robustness_lines(self, rob_df):
        perts = [p for p in PERTURBATIONS if p in set(rob_df.perturbation)]
        ncol = 4; nrow = math.ceil((len(perts) + 1) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow), sharey=True)
        axes = axes.ravel()
        clean = {m: rob_df[(rob_df.model == m) & (rob_df.perturbation == 'clean')]['Accuracy'].mean() for m in MODELS}
        for ax, p in zip(axes, perts):
            lv = sorted(rob_df[rob_df.perturbation == p].level.unique())
            vals = list(rob_df[(rob_df.perturbation == p) & (rob_df.level == lv[0])].level_value.iloc[:1])
            for m in MODELS:
                g = rob_df[(rob_df.model == m) & (rob_df.perturbation == p)].groupby('level')['Accuracy'].agg(['mean', 'std']).fillna(0)
                xs = [0] + list(g.index); ys = [clean[m]] + list(g['mean']); es = [0] + list(g['std'])
                ax.errorbar(xs, ys, yerr=es, color=MODEL_COLORS[m], lw=2.8 if m == 'Ensemble' else 1.4, marker='o', ms=5,
                            capsize=2, alpha=1 if m == 'Ensemble' else 0.7, label=m)
            ax.set_xticks([0] + lv); ax.set_xticklabels(['clean'] + [f'L{l}' for l in lv], fontsize=9)
            ax.set_title(p.replace('_', ' '), fontsize=11, fontweight='bold'); ax.grid(True)
        for ax in axes[len(perts):]:
            ax.axis('off')
        axes[0].set_ylabel('Accuracy'); axes[ncol].set_ylabel('Accuracy') if len(axes) > ncol else None
        h, l = axes[0].get_legend_handles_labels()
        axes[len(perts)].legend(h, l, loc='center', fontsize=12, frameon=False)
        self._finish(fig, 'robustness_accuracy_vs_severity')

    def robustness_heatmap(self, rob_df, col='Accuracy', as_drop=True, slug='robustness_heatmap_acc_drop', model='Ensemble'):
        perts = [p for p in PERTURBATIONS if p in set(rob_df.perturbation)] + [a for a in ABLATIONS if a in set(rob_df.perturbation)]
        d = rob_df[rob_df.model == model]
        clean = d[d.perturbation == 'clean'][col].mean()
        M = np.full((len(perts), 3), np.nan)
        for i, p in enumerate(perts):
            g = d[d.perturbation == p].groupby('level')[col].mean()
            for l, v in g.items():
                if 1 <= l <= 3:
                    M[i, l - 1] = (clean - v) * 100 if as_drop else v
        fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(perts) + 2))
        sns.heatmap(M, annot=True, fmt='.1f' if as_drop else '.3f', cmap='Reds' if as_drop else 'viridis', ax=ax,
                    xticklabels=['L1 (mild)', 'L2', 'L3 (severe)'], yticklabels=[p.replace('_', ' ') for p in perts],
                    cbar_kws={'label': f'{col} drop (percentage points)' if as_drop else col}, linewidths=0.8, linecolor='white')
        self._style(ax, None, None, f'{model}: {col} drop vs. clean (clean = {clean:.4f})' if as_drop else f'{model}: {col}', grid=False)
        self._finish(fig, slug)

    def robustness_retention(self, rob_df):
        d = rob_df[(rob_df.model == 'Ensemble') & (rob_df.perturbation != 'clean')]
        g = d.groupby('perturbation')['acc_retention_%'].agg(['mean', 'min']).sort_values('mean')
        fig, ax = plt.subplots(figsize=(8.0, 0.36 * len(g) + 2))
        yy = np.arange(len(g))
        ax.barh(yy, g['mean'], color=C['blue'], edgecolor='k', label='mean over severities')
        ax.plot(g['min'], yy, 'D', color=C['red'], mec='k', label='worst severity')
        ax.axvline(100, color='k', lw=1)
        ax.set_yticks(yy); ax.set_yticklabels([p.replace('_', ' ') for p in g.index])
        ax.set_xlim(max(0, g['min'].min() - 5), 105)
        self._style(ax, 'Accuracy retention (% of clean accuracy)', None, 'Ensemble robustness ranking')
        ax.legend(loc='lower right')
        self._finish(fig, 'robustness_retention')

    def robustness_by_model(self, rob_df):
        perts = [p for p in PERTURBATIONS if p in set(rob_df.perturbation)]
        top = {p: rob_df[rob_df.perturbation == p].level.max() for p in perts}
        fig, ax = plt.subplots(figsize=(12.5, 5.6))
        x = np.arange(len(perts)); w = 0.2
        for i, m in enumerate(MODELS):
            vals = [rob_df[(rob_df.model == m) & (rob_df.perturbation == p) & (rob_df.level == top[p])]['Accuracy'].mean() for p in perts]
            ax.bar(x + (i - 1.5) * w, vals, w, color=MODEL_COLORS[m], edgecolor='k', label=m)
        ax.set_xticks(x); ax.set_xticklabels([p.replace('_', '\n') for p in perts], fontsize=8.5)
        self._style(ax, None, 'Accuracy at the most severe level', 'Which branch breaks under which perturbation?')
        ax.legend(ncol=4, loc='lower left')
        self._finish(fig, 'robustness_per_branch')

    def augmentation_table(self, inv):
        d = inv[['augmentation', 'modality', 'stage', 'levels', 'unit']].copy()
        d['augmentation'] = d['augmentation'].str.slice(0, 34)
        d = d.set_index('augmentation')
        fig, ax = plt.subplots(figsize=(11, 0.3 * len(d) + 1.5)); ax.axis('off')
        tb = ax.table(cellText=d.values, rowLabels=d.index, colLabels=d.columns, cellLoc='left', loc='center')
        tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.25)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor('#BBBBBB')
            if r == 0: cell.set_facecolor(C['blue']); cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0: cell.set_facecolor('#F2F2F2')
        self._finish(fig, 'table_augmentations')

    # ---------- orchestration ----------
    def plot_reproducibility(self, df_seed, summ, rep_df):
        self.group('R_')
        for fn, a in [(self.seed_bars, (df_seed,)),
                      (lambda: self.curves_over_seeds('val_acc', 'Validation accuracy', 'val_accuracy_across_seeds', 'Validation accuracy across seeds'), ()),
                      (lambda: self.curves_over_seeds('val_loss', 'Validation loss', 'val_loss_across_seeds', 'Validation loss across seeds'), ()),
                      (self.roc_over_seeds, ()), (self.repro_table, (summ,)), (self.repeat_table, (rep_df,))]:
            self._safe(fn, *a)

    def plot_thresholds(self, thr_df, y, p, thr_map):
        self.group('T_')
        for fn, a in [(self.threshold_bars, (thr_df,)), (self.threshold_confusions, (y, p, thr_map)),
                      (self.threshold_roc, (y, p, thr_map)), (self.threshold_stability, (thr_df,)), (self.threshold_table, (thr_df,))]:
            self._safe(fn, *a)

    def plot_weights(self, w_df, gdf, y, probs, primary):
        self.group('W_')
        wp = w_df[w_df.seed == primary] if 'seed' in w_df else w_df
        for fn, a in [(self.weight_bars, (w_df,)), (self.weight_simplex, (gdf[gdf.seed == primary], wp)),
                      (self.weight_roc, (y, probs)), (self.weight_table, (w_df,))]:
            self._safe(fn, *a)

    def plot_significance(self, df_seed, sig_across, sig_seed, names, M, primary):
        self.group('S_')
        for fn, a in [(self.forest_across_seeds, (sig_across,)), (self.forest_per_seed, (sig_seed,)),
                      (self.pairwise_pvalues, (names, M, primary)), (self.seed_boxplots, (df_seed, sig_across)),
                      (self.sig_table, (sig_across,))]:
            self._safe(fn, *a)

    def plot_robustness(self, rob_df, inv):
        self.group('B_')
        for fn, a in [(self.augmentation_table, (inv,)), (self.robustness_lines, (rob_df,)),
                      (self.robustness_heatmap, (rob_df,)),
                      (lambda: self.robustness_heatmap(rob_df, 'AUC-ROC', False, 'robustness_heatmap_auc'), ()),
                      (self.robustness_retention, (rob_df,)), (self.robustness_by_model, (rob_df,))]:
            self._safe(fn, *a)

    def _safe(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:
            plt.close('all')
            name = getattr(fn, '__name__', 'figure')
            print(f"  ! {name} skipped: {type(e).__name__}: {e}")
# <<< PLOTTER2_END

# =====================================================================
# TRAINER
# =====================================================================

HISTORY_KEYS = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_auc', 'val_mcc', 'val_kappa',
                'lr', 'train_time', 'val_time', 'train_throughput']

class ImprovedTrainer:
    def __init__(self, model, save_dir):
        self.model = model.to(Config.DEVICE)
        self.device = Config.DEVICE
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        self.scaler = GradScaler(enabled=Config.USE_AMP)
        self.best_acc = 0.0
        self.history = {k: [] for k in HISTORY_KEYS}
        print(f"Trainer initialized on {self.device}")

    def _align_history(self):
        n = len(self.history['train_loss'])
        for k in HISTORY_KEYS:
            v = list(self.history.setdefault(k, []))
            self.history[k] = [np.nan] * (n - len(v)) + v if len(v) < n else v

    def _save_history(self):
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f)

    def train_model(self, train_loader, val_loader, resume_from=None):
        pre, new = [], []
        for name, p in self.model.named_parameters():
            (pre if ('feature_extractor' in name or 'video_features' in name) else new).append(p)
        optimizer = optim.AdamW([{'params': pre, 'lr': Config.BASE_LR * 0.1},
                                 {'params': new, 'lr': Config.BASE_LR}], weight_decay=Config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        patience_counter, start_epoch = 0, 0

        if resume_from and os.path.exists(resume_from):
            print(f"\nResuming from: {resume_from}")
            ck = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scheduler.load_state_dict(ck['scheduler_state_dict'])
            start_epoch = ck['epoch'] + 1
            self.best_acc = ck.get('best_acc', 0.0)
            patience_counter = ck.get('patience_counter', 0)
            self.history = ck.get('history', self.history)
            self._align_history()
            print(f"Resumed from epoch {ck['epoch']}, best acc: {self.best_acc:.4f}")

        for epoch in range(start_epoch, Config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}\n" + "-" * 60)

            cuda_sync(); t0 = time.perf_counter()
            tr_loss, tr_acc = self._train_epoch(train_loader, optimizer, scheduler, epoch)
            cuda_sync(); train_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            val = self.evaluate(val_loader, use_tta=(epoch > 30), desc="Validating")
            val_time = time.perf_counter() - t1
            vm = compute_all_metrics(val['labels'], val['probs'])

            for k, v in [('train_loss', tr_loss), ('train_acc', tr_acc), ('val_loss', val['loss']), ('val_acc', vm['Accuracy']),
                         ('val_f1', vm['F1-Score']), ('val_auc', vm.get('AUC-ROC', 0.0)), ('val_mcc', vm['MCC']),
                         ('val_kappa', vm["Cohen's Kappa"]), ('lr', optimizer.param_groups[1]['lr']),
                         ('train_time', train_time), ('val_time', val_time),
                         ('train_throughput', len(train_loader.dataset) / train_time)]:
                self.history[k].append(float(v))
            self._save_history()

            print(f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}  ({train_time:.1f}s, {len(train_loader.dataset) / train_time:.2f} vid/s)")
            print(f"Val   Loss: {val['loss']:.4f}, Acc: {vm['Accuracy']:.4f}  ({val_time:.1f}s)")
            kappa = vm["Cohen's Kappa"]
            print(f"Val F1: {vm['F1-Score']:.4f}  AUC: {vm.get('AUC-ROC', 0):.4f}  MCC: {vm['MCC']:.4f}  Kappa: {kappa:.4f}")

            def ckpt():
                return {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': self.best_acc, 'patience_counter': patience_counter,
                        'history': self.history, 'metrics': vm}

            if vm['Accuracy'] > self.best_acc + Config.MIN_DELTA:
                self.best_acc, patience_counter = vm['Accuracy'], 0
                torch.save(ckpt(), self.save_dir / 'best_model.pth')
                print(f"Best model saved! Acc: {self.best_acc:.4f}")
            else:
                patience_counter += 1
            torch.save(ckpt(), self.save_dir / 'latest_checkpoint.pth')

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        return self.best_acc

    def _train_epoch(self, dataloader, optimizer, scheduler, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        n_batches = len(dataloader)
        pbar = tqdm(dataloader, desc="Training")
        for i, (frames, audio, labels) in enumerate(pbar):
            frames = frames.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            if Config.USE_AMP:
                with amp_ctx():
                    outputs, _ = self.model(frames, audio)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs, _ = self.model(frames, audio)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
                optimizer.step()
            scheduler.step(epoch + (i + 1) / n_batches)   # T_0 / T_mult are in EPOCHS
            total_loss += loss.item()
            total += labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100. * correct / total:.2f}%'})
        return total_loss / n_batches, correct / total

    @torch.no_grad()
    def evaluate(self, loader, use_tta=False, desc="Evaluating", transform=None):
        """Returns labels, ensemble probs, per-model probs, loss and timing."""
        self.model.eval()
        n_iter = Config.TTA_ITERATIONS if (use_tta and Config.USE_TTA) else 1
        names = ['video', 'audio', 'fusion']
        labels_all, ens_all, comp_all = [], [], {k: [] for k in names}
        loss_sum, compute_time, n = 0.0, 0.0, 0
        wall0 = time.perf_counter()
        for frames, audio, labels in tqdm(loader, desc=desc, leave=False):
            frames = frames.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if transform is not None:                       # robustness / ablation hook (applied to test inputs only)
                frames, audio = transform(frames, audio)
            cuda_sync(); t0 = time.perf_counter()
            ens, comp = 0, {k: 0 for k in names}
            for it in range(n_iter):
                f, a = tta_transform(frames, audio, it) if n_iter > 1 else (frames, audio)
                with amp_ctx():
                    out, ind = self.model(f, a)
                ens = ens + F.softmax(out.float(), 1)
                for k in names:
                    comp[k] = comp[k] + F.softmax(ind[k].float(), 1)
            cuda_sync(); compute_time += time.perf_counter() - t0

            ens = ens / n_iter
            loss_sum += self.criterion(torch.log(ens + 1e-8), labels).item() * labels.size(0)
            n += labels.size(0)
            labels_all.append(labels.cpu().numpy()); ens_all.append(ens[:, 1].cpu().numpy())
            for k in names:
                comp_all[k].append((comp[k] / n_iter)[:, 1].cpu().numpy())
        wall = time.perf_counter() - wall0
        return {
            'labels': np.concatenate(labels_all), 'probs': np.concatenate(ens_all),
            'model_probs': {k: np.concatenate(v) for k, v in comp_all.items()},
            'loss': loss_sum / max(n, 1), 'wall_s': wall, 'compute_s': compute_time, 'n': n, 'tta_iters': n_iter,
        }

# =====================================================================
# >>> ROBUSTNESS_BEGIN
# Test-set robustness: every perturbation is applied to the TEST set only
# (the model is never re-trained / re-tuned on perturbed data)
# =====================================================================
import torchvision.transforms.functional as TF

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

def _denorm(f):
    return (f * _STD.to(f.device) + _MEAN.to(f.device)).clamp(0, 1)

def _renorm(p):
    return (p - _MEAN.to(p.device)) / _STD.to(p.device)

def _flat(f):
    return f.reshape(f.size(0) * f.size(1), *f.shape[2:])

def _unflat(x, B, T):
    return x.reshape(B, T, *x.shape[1:])

def _randn(shape, like, gen):
    return torch.randn(shape, generator=gen, device=like.device, dtype=like.dtype)

# ---- video perturbations: fn(frames[B,T,3,H,W] normalised, level, gen) -> frames ----
def _v_noise(f, s, gen):
    p = _denorm(f)
    return _renorm((p + _randn(p.shape, p, gen) * s).clamp(0, 1))

def _v_blur(f, s, gen):
    B, T = f.shape[:2]
    k = int(2 * math.ceil(3 * s) + 1)
    k = min(k, (min(f.shape[-2:]) // 2) * 2 - 1)
    return _unflat(TF.gaussian_blur(_flat(f), kernel_size=[k, k], sigma=[float(s), float(s)]), B, T)

def _v_bright(f, a, gen):
    return _renorm((_denorm(f) * a).clamp(0, 1))

def _v_contrast(f, a, gen):
    p = _denorm(f)
    m = p.mean(dim=(2, 3, 4), keepdim=True)
    return _renorm(((p - m) * a + m).clamp(0, 1))

def _v_jpeg(f, q, gen):
    B, T = f.shape[:2]
    arr = (_denorm(f) * 255).round().to(torch.uint8).permute(0, 1, 3, 4, 2).cpu().numpy()
    out = np.empty_like(arr)
    for b in range(B):
        for t in range(T):
            ok, enc = cv2.imencode('.jpg', arr[b, t][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(q)])
            out[b, t] = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
    p = torch.from_numpy(np.ascontiguousarray(out)).to(f.device).permute(0, 1, 4, 2, 3).float() / 255.0
    return _renorm(p)

def _v_lowres(f, s, gen):
    B, T = f.shape[:2]
    x = _flat(f)
    H, W = x.shape[-2:]
    d = F.interpolate(x, size=(max(2, int(H * s)), max(2, int(W * s))), mode='bilinear', align_corners=False, antialias=True)
    return _unflat(F.interpolate(d, size=(H, W), mode='bilinear', align_corners=False), B, T)

def _v_rotate(f, ang, gen):
    B, T = f.shape[:2]
    return _unflat(TF.rotate(_flat(f), float(ang), interpolation=TF.InterpolationMode.BILINEAR), B, T)

def _v_zoom(f, s, gen):
    B, T = f.shape[:2]
    x = _flat(f)
    H, W = x.shape[-2:]
    h2, w2 = max(2, int(H * s)), max(2, int(W * s))
    t0, l0 = (H - h2) // 2, (W - w2) // 2
    c = x[..., t0:t0 + h2, l0:l0 + w2]
    return _unflat(F.interpolate(c, size=(H, W), mode='bilinear', align_corners=False), B, T)

def _v_flip(f, _, gen):
    return torch.flip(f, dims=[-1])

def _v_framedrop(f, q, gen):
    B, T = f.shape[:2]
    keep = torch.rand(B, T, generator=gen, device=f.device) > q
    keep[:, 0] = True
    out = f.clone()
    for t in range(1, T):                                     # forward-fill dropped frames
        m = keep[:, t].view(B, 1, 1, 1)
        out[:, t] = torch.where(m, out[:, t], out[:, t - 1])
    return out

def _v_occlude(f, r, gen):
    B, T, C, H, W = f.shape
    h, w = max(1, int(H * r)), max(1, int(W * r))
    ys = torch.randint(0, H - h + 1, (B,), generator=gen, device=f.device)
    xs = torch.randint(0, W - w + 1, (B,), generator=gen, device=f.device)
    out = f.clone()
    for b in range(B):
        y0, x0 = int(ys[b]), int(xs[b])
        out[b, :, :, y0:y0 + h, x0:x0 + w] = 0.0              # 0 in normalised space = dataset-mean colour
    return out

# ---- audio perturbations: fn(mel[B,1,F,T] standardised, level, gen) -> mel ----
def _a_noise(a, s, gen):
    return a + _randn(a.shape, a, gen) * s

def _a_timemask(a, r, gen):
    B, _, Fq, T = a.shape
    w = max(1, int(T * r))
    st = torch.randint(0, max(1, T - w + 1), (B,), generator=gen, device=a.device)
    out = a.clone()
    for b in range(B):
        out[b, :, :, int(st[b]):int(st[b]) + w] = 0.0
    return out

def _a_freqmask(a, r, gen):
    B, _, Fq, T = a.shape
    w = max(1, int(Fq * r))
    st = torch.randint(0, max(1, Fq - w + 1), (B,), generator=gen, device=a.device)
    out = a.clone()
    for b in range(B):
        out[b, :, int(st[b]):int(st[b]) + w, :] = 0.0
    return out

# name -> spec. 'levels' are the 3 severities (mild -> severe)
PERTURBATIONS = {
    'gaussian_noise':   dict(modality='video', fn=_v_noise,     levels=[0.02, 0.05, 0.10], unit='sigma (pixel units)'),
    'gaussian_blur':    dict(modality='video', fn=_v_blur,      levels=[1.0, 2.0, 4.0],    unit='sigma (px)'),
    'brightness_down':  dict(modality='video', fn=_v_bright,    levels=[0.7, 0.5, 0.3],    unit='factor'),
    'brightness_up':    dict(modality='video', fn=_v_bright,    levels=[1.3, 1.6, 2.0],    unit='factor'),
    'contrast_down':    dict(modality='video', fn=_v_contrast,  levels=[0.7, 0.5, 0.3],    unit='factor'),
    'jpeg_compression': dict(modality='video', fn=_v_jpeg,      levels=[60, 30, 10],       unit='JPEG quality'),
    'low_resolution':   dict(modality='video', fn=_v_lowres,    levels=[0.5, 0.25, 0.125], unit='down-scale factor'),
    'rotation':         dict(modality='video', fn=_v_rotate,    levels=[5, 15, 30],        unit='degrees'),
    'center_zoom':      dict(modality='video', fn=_v_zoom,      levels=[0.9, 0.75, 0.6],   unit='kept fraction'),
    'horizontal_flip':  dict(modality='video', fn=_v_flip,      levels=[1],                unit='-'),
    'frame_drop':       dict(modality='video', fn=_v_framedrop, levels=[0.1, 0.3, 0.5],    unit='drop rate'),
    'occlusion_patch':  dict(modality='video', fn=_v_occlude,   levels=[0.2, 0.35, 0.5],   unit='patch side / frame side'),
    'audio_noise':      dict(modality='audio', fn=_a_noise,     levels=[0.1, 0.3, 0.6],    unit='sigma (std. mel units)'),
    'audio_time_mask':  dict(modality='audio', fn=_a_timemask,  levels=[0.1, 0.25, 0.5],   unit='masked fraction of time'),
    'audio_freq_mask':  dict(modality='audio', fn=_a_freqmask,  levels=[0.1, 0.25, 0.5],   unit='masked fraction of mel bins'),
}

# modality drop-out ablation: the whole input of one modality is replaced by the dataset mean
ABLATIONS = {
    'no_audio (audio zeroed)': lambda f, a: (f, torch.zeros_like(a)),
    'no_video (frames zeroed)': lambda f, a: (torch.zeros_like(f), a),
}

def make_transform(name, level, seed, device):
    spec = PERTURBATIONS[name]
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    def tf(frames, audio):
        if spec['modality'] == 'video':
            return spec['fn'](frames, level, gen), audio
        return frames, spec['fn'](audio, level, gen)
    return tf

ROB_KEYS = ['Accuracy', 'Balanced Acc.', 'Precision', 'Recall', 'F1-Score', 'MCC', "Cohen's Kappa", 'AUC-ROC', 'FPR', 'FNR']

def augmentation_inventory():
    """Everything that is applied: training augmentations + test-time robustness perturbations."""
    rows = []
    p = Config.AUGMENTATION_PROB
    train = [
        ('Horizontal flip', 'video', 'training', p * 0.5, 'all frames of a clip flipped together'),
        ('Brightness scaling', 'video', 'training', p * 0.5, 'factor ~ U(0.8, 1.2), clamp [-2.5, 2.5]'),
        ('Colour shift', 'video', 'training', p * 0.3, 'per-channel offset ~ N(0, 0.1^2)'),
        ('Frame dropout (temporal)', 'video', 'training', p * 0.6 * 0.5, '~10% frames replaced by first kept frame'),
        ('Temporal jitter / shuffle', 'video', 'training', p * 0.6 * 0.3, 'random permutation shifted by +-3 frames'),
        ('Gaussian noise', 'audio', 'training', p * 0.5, 'sigma = 0.01 on standardised mel'),
        ('Time masking', 'audio', 'training', p * 0.3, 'up to 10% of time axis'),
        ('Frequency masking', 'audio', 'training', p * 0.3, 'up to 10% of mel bins'),
        ('TTA: identity / h-flip / brightness x1.1', 'video', 'test-time (TTA)', 1.0, f'{Config.TTA_ITERATIONS} passes averaged'),
    ]
    for n, mod, stage, prob, desc in train:
        rows.append({'augmentation': n, 'modality': mod, 'stage': stage, 'approx_probability': prob,
                     'levels': '-', 'unit': '-', 'description': desc})
    for n, s in PERTURBATIONS.items():
        rows.append({'augmentation': n, 'modality': s['modality'], 'stage': 'test-set robustness',
                     'approx_probability': 1.0, 'levels': ', '.join(map(str, s['levels'])), 'unit': s['unit'],
                     'description': 'applied to every test video; model not re-trained'})
    for n in ABLATIONS:
        rows.append({'augmentation': n, 'modality': 'ablation', 'stage': 'test-set ablation', 'approx_probability': 1.0,
                     'levels': '1', 'unit': '-', 'description': 'modality removed at inference'})
    return pd.DataFrame(rows)

def run_robustness(trainer, test_loader, seed, clean=None):
    """Evaluate all perturbations on the test set. Returns a long DataFrame
    (seed, perturbation, modality, level, value, model, <metrics>)."""
    dev = trainer.device
    rows = []

    def add(res, pert, modality, lvl, val, unit):
        y = res['labels']
        probs = {'Video': res['model_probs']['video'], 'Audio': res['model_probs']['audio'],
                 'Fusion': res['model_probs']['fusion'], 'Ensemble': res['probs']}
        for mn, p in probs.items():
            m = compute_all_metrics(y, p)
            rows.append({'seed': seed, 'perturbation': pert, 'modality': modality, 'level': lvl, 'level_value': val,
                         'unit': unit, 'model': mn, **{k: m[k] for k in ROB_KEYS}})

    clean_res = clean or trainer.evaluate(test_loader, use_tta=Config.ROBUSTNESS_USE_TTA, desc="Robustness: clean")
    add(clean_res, 'clean', '-', 0, 0, '-')

    total = sum(len(s['levels']) for s in PERTURBATIONS.values()) + len(ABLATIONS)
    done = 0
    for name, spec in PERTURBATIONS.items():
        for li, val in enumerate(spec['levels'], start=1):
            done += 1
            res = trainer.evaluate(test_loader, use_tta=Config.ROBUSTNESS_USE_TTA,
                                   transform=make_transform(name, val, seed * 1000 + li, dev),
                                   desc=f"Robustness [{done}/{total}] {name} L{li}")
            add(res, name, spec['modality'], li, val, spec['unit'])
    for name, fn in ABLATIONS.items():
        done += 1
        res = trainer.evaluate(test_loader, use_tta=Config.ROBUSTNESS_USE_TTA, transform=fn,
                               desc=f"Ablation [{done}/{total}] {name}")
        add(res, name, 'ablation', 1, 1, '-')

    df = pd.DataFrame(rows)
    base = df[(df.perturbation == 'clean')].set_index('model')
    df['acc_drop'] = df.apply(lambda r: base.loc[r.model, 'Accuracy'] - r['Accuracy'], axis=1)
    df['acc_retention_%'] = df.apply(lambda r: 100 * r['Accuracy'] / max(base.loc[r.model, 'Accuracy'], EPS), axis=1)
    df['auc_drop'] = df.apply(lambda r: base.loc[r.model, 'AUC-ROC'] - r['AUC-ROC'], axis=1)
    return df

def robustness_summary(df):
    """One row per (perturbation, model): mean over severities of accuracy / drop / AUC, worst case."""
    d = df[~df.perturbation.isin(['clean'])]
    g = d.groupby(['perturbation', 'modality', 'model'], sort=False)
    out = g.agg(mean_accuracy=('Accuracy', 'mean'), worst_accuracy=('Accuracy', 'min'), mean_auc=('AUC-ROC', 'mean'),
                mean_f1=('F1-Score', 'mean'), mean_mcc=('MCC', 'mean'), mean_acc_drop=('acc_drop', 'mean'),
                max_acc_drop=('acc_drop', 'max'), mean_retention_pct=('acc_retention_%', 'mean')).reset_index()
    return out
# <<< ROBUSTNESS_END

# =====================================================================
# REPORTING
# =====================================================================

REPORT_KEYS = ['Accuracy', 'Balanced Acc.', 'Precision', 'Recall', 'Specificity', 'NPV', 'F1-Score', 'F2-Score', 'Macro-F1',
               'MCC', "Cohen's Kappa", 'G-Mean', "Youden's J", 'Markedness', 'Fowlkes-Mallows', 'Jaccard', 'FPR', 'FNR',
               'FDR', 'FOR', 'LR+', 'LR-', 'AUC-ROC', 'AP (AUC-PR)', 'pAUC@FPR<=0.1', 'EER', 'KS Statistic',
               'TPR@FPR=1%', 'TPR@FPR=0.1%', 'Log Loss', 'Brier Score', 'ECE', 'TP', 'TN', 'FP', 'FN']

def print_report(R):
    df = pd.DataFrame({n: {k: R['metrics'][n].get(k, np.nan) for k in REPORT_KEYS} for n in MODELS})
    df['Ensemble 95% CI'] = [f"[{R['ci'][k][0]:.4f}, {R['ci'][k][1]:.4f}]" if k in R['ci'] else '' for k in REPORT_KEYS]
    print("\n" + "=" * 90 + f"\nTEST-SET METRICS (seed {R['seed']})\n" + "=" * 90)
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))
    T = R['timing']
    print("\n" + "=" * 90 + "\nTIMING & EFFICIENCY\n" + "=" * 90)
    print(f"Total training time        : {fmt_time(T['training_total_s'])}  over {T['n_epochs']} epochs")
    print(f"Mean train epoch / val     : {T['train_epoch_mean_s']:.1f} s / {T['val_epoch_mean_s']:.1f} s")
    print(f"Testing time (wall, TTAx{T['tta_iters']})  : {fmt_time(T['test_wall_s'])} for {T['test_samples']} videos")
    print(f"  model compute / data I/O : {T['test_compute_s']:.2f} s / {T['test_data_s']:.2f} s")
    print(f"  end-to-end throughput    : {T['test_throughput_e2e']:.2f} videos/s")
    print(f"  model-only throughput    : {T['test_throughput_model']:.2f} videos/s")
    if R.get('bench'):
        s = R['bench']['single_ms']['Ensemble']
        print(f"Latency bs=1 (ensemble)    : {s.mean():.2f} ± {s.std():.2f} ms | P50 {np.percentile(s, 50):.2f} | "
              f"P95 {np.percentile(s, 95):.2f} | P99 {np.percentile(s, 99):.2f}")
        for bs, v in R['bench']['sweep'].items():
            print(f"  bs={bs:<3d}: {v['throughput_vps']:8.2f} videos/s  {v['throughput_fps']:9.1f} frames/s  {v['ms_per_video']:7.2f} ms/video")
        print(f"Peak GPU memory            : {R['bench']['peak_gpu_mem_MB']:.0f} MB")
        for n, c in R['bench']['complexity'].items():
            print(f"  {n:<9s}: {c['params_M']:7.2f} M params | {c['size_MB']:7.1f} MB | {c['GFLOPs']:8.2f} GFLOPs")
    return df

def export_results(R, df, out_dir, prefix=''):
    out = Path(out_dir)
    df.to_csv(out / f'{prefix}metrics_summary.csv')
    try:
        df.drop(columns=['Ensemble 95% CI']).to_latex(out / f'{prefix}metrics_summary.tex', float_format="%.4f")
    except Exception as e:
        print(f"(LaTeX export skipped: {e})")
    pd.DataFrame(R['history']).to_csv(out / f'{prefix}per_epoch_history.csv', index_label='epoch')
    payload = {'seed': R['seed'], 'metrics': R['metrics'], 'ci_95': R['ci'], 'timing': R['timing']}
    if R.get('bench'):
        payload['efficiency'] = {'sweep': R['bench']['sweep'], 'complexity': R['bench']['complexity'],
                                 'peak_gpu_mem_MB': R['bench']['peak_gpu_mem_MB'],
                                 'latency_ms_bs1': {k: {'mean': float(v.mean()), 'std': float(v.std()),
                                                        'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)),
                                                        'p99': float(np.percentile(v, 99))}
                                                    for k, v in R['bench']['single_ms'].items()}}
        rows = [{'batch_size': bs, **v} for bs, v in R['bench']['sweep'].items()]
        pd.DataFrame(rows).to_csv(out / f'{prefix}efficiency_throughput_vs_batch.csv', index=False)
        pd.DataFrame(R['bench']['complexity']).T.to_csv(out / f'{prefix}efficiency_complexity.csv', index_label='model')
        pd.DataFrame({k: pd.Series(v) for k, v in R['bench']['single_ms'].items()}).to_csv(out / f'{prefix}latency_samples_ms.csv', index=False)
    with open(out / f'{prefix}final_results.json', 'w') as f:
        json.dump(to_serializable(payload), f, indent=2)

# =====================================================================
# PER-SEED PIPELINE  (train / resume / skip -> evaluate -> optional bench / robustness)
# =====================================================================

def _pack(res):
    return {'Video': res['model_probs']['video'], 'Audio': res['model_probs']['audio'],
            'Fusion': res['model_probs']['fusion'], 'Ensemble': res['probs']}

def save_predictions(ds, y, probs, path):
    thr = 0.5
    pd.DataFrame({'video': [os.path.basename(s[0]) for s in ds.samples], 'path': [s[0] for s in ds.samples],
                  'label': y, 'p_video': probs['Video'], 'p_audio': probs['Audio'], 'p_fusion': probs['Fusion'],
                  'p_ensemble': probs['Ensemble'], 'pred@0.5': (probs['Ensemble'] >= thr).astype(int)}).to_csv(path, index=False)

def run_seed_pipeline(seed, train_ds, val_ds, test_ds, val_loader, test_loader, tag='',
                      do_bench=False, do_robust=False):
    import gc
    sdir = Path(Config.OUTPUT_DIR) / f"seed_{seed}{tag}"
    sdir.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 78 + f"\n#  SEED {seed}{tag}   ->   {sdir}\n" + "#" * 78)

    set_seed(seed)
    model = ImprovedEnsembleDetector()
    trainer = ImprovedTrainer(model, sdir)
    done_flag, best_path = sdir / 'train_done.json', sdir / 'best_model.pth'

    if Config.SKIP_FINISHED_TRAINING and done_flag.exists() and best_path.exists() and (sdir / 'history.json').exists():
        print("Training already finished for this seed -> loading best checkpoint (delete train_done.json to retrain).")
        with open(sdir / 'history.json') as f:
            trainer.history = json.load(f)
    else:
        g = torch.Generator(); g.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0,
                                  pin_memory=True, drop_last=True, generator=g)
        latest = sdir / 'latest_checkpoint.pth'
        resume = str(latest) if (Config.AUTO_RESUME and latest.exists()) else None
        t0 = time.time()
        trainer.train_model(train_loader, val_loader, resume_from=resume)
        with open(done_flag, 'w') as f:
            json.dump({'seed': seed, 'best_val_acc': trainer.best_acc, 'epochs_run': len(trainer.history['train_loss']),
                       'last_session_hours': (time.time() - t0) / 3600}, f, indent=2)

    ck = torch.load(best_path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    history = trainer.history
    best_epoch = int(ck['epoch']) + 1

    print("\nEvaluating best model: validation set (used ONLY to choose thresholds / weights)...")
    val = trainer.evaluate(val_loader, use_tta=True, desc="Validation (final)")          # also warms up the GPU
    print("Evaluating best model: TEST set...")
    test = trainer.evaluate(test_loader, use_tta=True, desc="Testing")
    y = test['labels']
    probs, val_probs = _pack(test), _pack(val)

    # self-consistency: ensemble prob must equal the weighted sum of component probs
    if not np.allclose(ens_from_weights(probs, Config.ENSEMBLE_WEIGHTS), probs['Ensemble'], atol=5e-3):
        print("  ! WARNING: stored ensemble probability deviates from the weighted component probabilities "
              "(weight-comparison results may be slightly off). Check Config.ENSEMBLE_WEIGHTS.")

    # repeatability of inference (same weights, same data, repeated)
    rep_df = pd.DataFrame()
    if Config.REPEAT_EVAL_RUNS > 1:
        runs = [test['probs']] + [trainer.evaluate(test_loader, use_tta=True, desc=f"Repeat {i}")['probs']
                                  for i in range(1, Config.REPEAT_EVAL_RUNS)]
        rep_df = repeatability_check(runs)
        rep_df.insert(0, 'seed', seed)

    metrics = {n: compute_all_metrics(y, p) for n, p in probs.items()}
    print(f"Bootstrapping {Config.BOOTSTRAP_ITERS} resamples for 95% CIs...")
    boot, ci = bootstrap_metrics(y, probs['Ensemble'])
    save_predictions(test_ds, y, probs, sdir / 'test_predictions.csv')
    save_predictions(val_ds, val['labels'], val_probs, sdir / 'val_predictions.csv')

    et = np.nan_to_num(np.asarray(history['train_time'], float)) + np.nan_to_num(np.asarray(history['val_time'], float))
    timing = {
        'training_total_s': float(np.nansum(et)), 'n_epochs': len(history['train_loss']), 'best_epoch': best_epoch,
        'train_epoch_mean_s': float(np.nanmean(history['train_time'])) if len(history['train_time']) else float('nan'),
        'val_epoch_mean_s': float(np.nanmean(history['val_time'])) if len(history['val_time']) else float('nan'),
        'test_wall_s': test['wall_s'], 'test_compute_s': test['compute_s'], 'test_data_s': test['wall_s'] - test['compute_s'],
        'test_samples': test['n'], 'tta_iters': test['tta_iters'],
        'test_throughput_e2e': test['n'] / test['wall_s'], 'test_throughput_model': test['n'] / test['compute_s'],
        'test_latency_ms_per_video': test['compute_s'] / test['n'] * 1000,
    }

    bench = None
    if do_bench:
        print("Benchmarking inference efficiency...")
        bench = run_efficiency_benchmark(model, test_loader)

    rob_df = None
    if do_robust:
        rob_csv = sdir / 'robustness_results.csv'
        if Config.REUSE_CACHED_ROBUSTNESS and rob_csv.exists():
            print(f"Re-using cached robustness results: {rob_csv}")
            rob_df = pd.read_csv(rob_csv)
        else:
            print("\nRobustness evaluation on the test set (perturbations / augmentations)...")
            rob_df = run_robustness(trainer, test_loader, seed)
            rob_df.to_csv(rob_csv, index=False)

    S = {'seed': seed, 'tag': tag, 'y_true': y, 'probs': probs, 'metrics': metrics, 'ci': ci, 'boot': boot,
         'timing': timing, 'bench': bench, 'history': history, 'val': {'y': val['labels'], 'probs': val_probs},
         'repeat': rep_df, 'robust': rob_df}
    del model, trainer
    gc.collect()
    if Config.DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return S

# =====================================================================
# CROSS-SEED ANALYSIS, CSV EXPORT AND FIGURES
# =====================================================================

def analyze_and_report(per_seed, repeat_train=None):
    out = Path(Config.OUTPUT_DIR)
    csv_dir = out / 'csv'; csv_dir.mkdir(exist_ok=True)
    primary = Config.PRIMARY_SEED if Config.PRIMARY_SEED in per_seed else next(iter(per_seed))
    Rp = per_seed[primary]
    saved = {}

    def save(df, name, index=False):
        df.to_csv(csv_dir / name, index=index)
        saved[name] = len(df)

    # ---- 1. primary-seed deep dive (full metric table, timing, efficiency) ----
    df_primary = print_report(Rp)
    export_results(Rp, df_primary, csv_dir, prefix=f'seed{primary}_')

    # ---- 2. reproducibility ----
    df_seed = per_seed_table(per_seed)
    save(df_seed, 'per_seed_metrics.csv')
    summ = reproducibility_summary(df_seed)
    save(summ, 'reproducibility_summary.csv')
    save(summ.pivot(index='metric', columns='model', values='mean_pm_std').reindex(REPRO_KEYS), 'reproducibility_mean_pm_std.csv', index=True)
    save(pd.DataFrame([{'seed': s, **{f'{k}': v for k, v in S['timing'].items()}} for s, S in per_seed.items()]), 'timing_per_seed.csv')
    rep_all = pd.concat([S['repeat'] for S in per_seed.values() if len(S['repeat'])], ignore_index=True) if any(len(S['repeat']) for S in per_seed.values()) else pd.DataFrame()
    if len(rep_all): save(rep_all, 'repeatability_inference.csv')
    if repeat_train is not None:
        save(repeat_train, 'repeatability_training.csv')
    print("\n" + "=" * 90 + f"\nREPRODUCIBILITY over seeds {list(per_seed)} (Ensemble)\n" + "=" * 90)
    print(summ[summ.model == 'Ensemble'][['metric', 'mean_pm_std', 'ci95_low', 'ci95_high', 'min', 'max', 'cv_percent']]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 3. four thresholds (chosen on validation, evaluated on test) ----
    thr_df = pd.concat([threshold_comparison(S['val']['y'], S['val']['probs']['Ensemble'], S['y_true'], S['probs']['Ensemble'], seed=s)
                        for s, S in per_seed.items()], ignore_index=True)
    save(thr_df, 'threshold_comparison_per_seed.csv')
    thr_keys = ['threshold', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC', "Cohen's Kappa", 'AUC-ROC', 'FPR', 'FNR']
    save(aggregate_over_seeds(thr_df, 'strategy', thr_keys), 'threshold_comparison_summary.csv')
    print("\n" + "=" * 90 + "\nTHRESHOLD COMPARISON (mean over seeds)\n" + "=" * 90)
    print(thr_df.groupby('strategy', sort=False)[thr_keys].mean().to_string(float_format=lambda v: f"{v:.4f}"))
    thr_map = select_thresholds(Rp['val']['y'], Rp['val']['probs']['Ensemble'])

    # ---- 4. ensemble weights ----
    w_list, g_list, w_probs_primary = [], [], None
    for s, S in per_seed.items():
        wdf, gdf, wp = weight_comparison(S['val']['probs'], S['val']['y'], S['probs'], S['y_true'], seed=s)
        w_list.append(wdf); g_list.append(gdf)
        if s == primary: w_probs_primary = wp
    w_df, g_df = pd.concat(w_list, ignore_index=True), pd.concat(g_list, ignore_index=True)
    save(w_df, 'ensemble_weights_per_seed.csv')
    save(aggregate_over_seeds(w_df, 'config', ['w_video', 'w_audio', 'w_fusion', 'Accuracy', 'F1-Score', 'AUC-ROC', 'MCC', "Cohen's Kappa"]),
         'ensemble_weights_summary.csv')
    save(g_df, 'ensemble_weight_simplex_grid.csv')
    print("\n" + "=" * 90 + "\nENSEMBLE-WEIGHT COMPARISON (mean over seeds)\n" + "=" * 90)
    print(w_df.groupby('config', sort=False)[['w_video', 'w_audio', 'w_fusion', 'Accuracy', 'F1-Score', 'AUC-ROC', 'MCC']].mean()
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- 5. statistical significance ----
    sig_seed = pd.concat([significance_within_seed(S['y_true'], S['probs'], seed=s) for s, S in per_seed.items()], ignore_index=True)
    save(sig_seed, 'significance_per_seed_mcnemar_delong_bootstrap.csv')
    sig_across, sig_target = significance_across_seeds(df_seed)
    save(sig_across, 'significance_across_seeds.csv')
    save(sig_target, 'significance_vs_90pct_target.csv')
    names, M = pairwise_mcnemar_matrix(Rp['y_true'], Rp['probs'])
    save(pd.DataFrame(M, index=names, columns=names), f'pairwise_mcnemar_pvalues_seed{primary}.csv', index=True)
    print("\n" + "=" * 90 + "\nSIGNIFICANCE: Ensemble vs components (across seeds, Holm-adjusted)\n" + "=" * 90)
    print(sig_across[['metric', 'B', 'mean_diff', 'cohens_dz', 'paired_t_p', 'paired_t_p_holm', 't_sig_holm', 'wilcoxon_p']]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 6. robustness / augmentation ----
    inv = augmentation_inventory()
    save(inv, 'augmentation_inventory.csv')
    rob_all = pd.concat([S['robust'] for S in per_seed.values() if S['robust'] is not None], ignore_index=True) \
        if any(S['robust'] is not None for S in per_seed.values()) else None
    if rob_all is not None:
        save(rob_all, 'robustness_results.csv')
        save(robustness_summary(rob_all), 'robustness_summary.csv')
        piv = rob_all[rob_all.model == 'Ensemble'].pivot_table(index='perturbation', columns='level', values='Accuracy', aggfunc='mean')
        save(piv, 'robustness_ensemble_accuracy_pivot.csv', index=True)
        print("\n" + "=" * 90 + "\nROBUSTNESS: Ensemble accuracy by perturbation x severity\n" + "=" * 90)
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- 7. figures: display -> 600 dpi ----
    ResearchPlotter(out, Rp).plot_all()
    AP = AnalysisPlotter(out, per_seed, primary)
    AP.plot_reproducibility(df_seed, summ, rep_all)
    AP.plot_thresholds(thr_df, Rp['y_true'], Rp['probs']['Ensemble'], thr_map)
    AP.plot_weights(w_df, g_df, Rp['y_true'], w_probs_primary, primary)
    AP.plot_significance(df_seed, sig_across, sig_seed, names, M, primary)
    if rob_all is not None:
        AP.plot_robustness(rob_all, inv)

    manifest = pd.DataFrame({'csv_file': list(saved), 'rows': list(saved.values())})
    manifest.to_csv(csv_dir / '00_manifest.csv', index=False)
    print(f"\n✓ {len(saved)} CSV files in {csv_dir}")
    print(f"✓ figures in {out / f'figures_{Config.DPI}dpi'}")

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("PRETRAINED ENSEMBLE DEEPFAKE DETECTION - RESEARCH EDITION (multi-seed)\n" + "=" * 70)
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount('/content/drive')
    except Exception:
        pass
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print("\nLoading datasets (fixed, disjoint split for ALL seeds -> paired statistics are valid)...")
    train_ds = ImprovedDeepfakeDataset(Config.DATASET_PATH, "train", Config.CACHE_DATA)
    val_ds = ImprovedDeepfakeDataset(Config.DATASET_PATH, "val", Config.CACHE_DATA)
    test_ds = ImprovedDeepfakeDataset(Config.DATASET_PATH, "test", Config.CACHE_DATA)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    seeds = list(Config.SEEDS)
    primary = Config.PRIMARY_SEED if Config.PRIMARY_SEED in seeds else seeds[0]
    per_seed = {}
    for seed in seeds:
        per_seed[seed] = run_seed_pipeline(
            seed, train_ds, val_ds, test_ds, val_loader, test_loader,
            do_bench=(seed == primary),
            do_robust=(Config.RUN_ROBUSTNESS and seed in Config.ROBUSTNESS_SEEDS))

    repeat_train = None
    if Config.REPEAT_TRAIN_CHECK:                       # train the primary seed a 2nd time from scratch
        print("\nREPEATABILITY OF TRAINING: re-training the primary seed from scratch...")
        S2 = run_seed_pipeline(primary, train_ds, val_ds, test_ds, val_loader, test_loader, tag='_repeat')
        a, b = per_seed[primary], S2
        d = np.abs(a['probs']['Ensemble'] - b['probs']['Ensemble'])
        repeat_train = pd.DataFrame([{
            'seed': primary, 'acc_run1': a['metrics']['Ensemble']['Accuracy'], 'acc_run2': b['metrics']['Ensemble']['Accuracy'],
            'auc_run1': a['metrics']['Ensemble']['AUC-ROC'], 'auc_run2': b['metrics']['Ensemble']['AUC-ROC'],
            'max_abs_prob_diff': float(d.max()), 'mean_abs_prob_diff': float(d.mean()),
            'identical_predictions_%': float(((a['probs']['Ensemble'] >= .5) == (b['probs']['Ensemble'] >= .5)).mean() * 100)}])

    analyze_and_report(per_seed, repeat_train)
    print(f"\nAll outputs in: {Config.OUTPUT_DIR}")
    return per_seed

if __name__ == "__main__":
    per_seed = main()
