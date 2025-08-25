import os
import pickle
import numpy as np
import librosa
import librosa.feature
import cv2
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
# ---------------------------
# 音频预处理与特征提取
# ---------------------------

def preprocess_spectrogram(S, size=(224, 224)):
    """
    将输入的特征图归一化到 0-255，并调整尺寸为 224×224，
    返回单通道图像
    """
    S_norm = 255.0 * (S - S.min()) / (S.max() - S.min() + 1e-6)
    S_resized = cv2.resize(S_norm.astype(np.uint8), size)
    return S_resized

def extract_features(y, sr, target_size=(224, 224)):
    """
    提取梅尔谱图、STFT、mfcc并融合成 3 通道特征。
    """
    # 提取梅尔谱图
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 提取 STFT 频谱
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

    # 提取 MFCC（使用 128 维）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, n_fft=2048, hop_length=512)

    # 统一尺寸
    mel_spec = preprocess_spectrogram(mel_spec,target_size)
    stft= preprocess_spectrogram(stft, target_size)
    mfcc = preprocess_spectrogram(mfcc, target_size)

    # 组合成 3 通道特征
    fused_features = np.stack([mel_spec, stft, mfcc], axis=-1)

    return fused_features

# ---------------------------
# 自定义数据集，增加缓存功能
# ---------------------------
class BirdsongDataset(Dataset):
    def __init__(self, data_dir, sr=48000, use_cache=True, cache_file="mel+mfcc+stft.pkl" ):
        """
        参数:
          data_dir: 存放数据的根目录，每个子文件夹对应一个类别
          sr: 采样率
          use_cache: 是否启用缓存，若为 True，则首次加载后保存缓存，后续直接加载缓存数据
          cache_file: 缓存文件名称，保存在 data_dir 下
        """
        self.data_dir = data_dir
        self.sr = sr
        self.cache_path = os.path.join(data_dir, cache_file)
        self.files = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}


        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for file in os.listdir(cls_dir):
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif')):
                    self.files.append(os.path.join(cls_dir, file))
                    self.labels.append(self.class_to_idx[cls])

        if use_cache and os.path.exists(self.cache_path):
            print("从缓存中加载特征数据...")
            with open(self.cache_path, 'rb') as f:
                self.features = pickle.load(f)
        else:
            print("未找到缓存，正在提取音频特征...")
            self.features = self.load_features()
            if use_cache:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.features, f)
                print(f"特征数据已保存到缓存: {self.cache_path}")

    def load_features(self):
        features = []
        with ThreadPoolExecutor(max_workers=24) as executor:
            results = list(tqdm(executor.map(self.process_file, self.files),
                                total=len(self.files), desc="Loading audio"))
        for res in results:
            features.extend(res)
        return features

    def process_file(self, file_path):
        """ 加载整个音频文件并提取特征 """
        y, sr = librosa.load(file_path, sr=self.sr)
        feat = extract_features(y, sr)
        return [feat]  # 以列表形式返回

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------
# 数据加载与划分
# ---------------------------
dataset = BirdsongDataset('ganzhou_38', use_cache=True)
# 按 8:2 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
all_indices = list(range(len(dataset)))
all_labels = dataset.labels
train_idx, val_idx= train_test_split(all_indices, test_size=0.2, stratify=all_labels,random_state=42)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=24)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=24)
print(f"训练集: {train_size}, 验证集: {val_size}")
