# Colab 학습 노트북 — NAFNet on SIDL (셀별 템플릿)

> 이 파일을 참고해 Colab 노트북을 만드세요. 각 섹션이 하나의 셀입니다.
> 첫 사용 시 **위에서 아래로 순서대로** 실행.

---

## 셀 1 — Drive mount + 폴더 준비

```python
from google.colab import drive
drive.mount('/content/drive')

import os
WORK = '/content/drive/MyDrive/SIDL'
DATA = f'{WORK}/data'
CKPT = f'{WORK}/checkpoints'
CODE = f'{WORK}/code'
for d in [WORK, DATA, CKPT, CODE]:
    os.makedirs(d, exist_ok=True)
print('Folders ready:', os.listdir(WORK))
```

---

## 셀 2 — GPU 확인

```python
import torch
print(f'Torch: {torch.__version__}')
print(f'CUDA:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
```

---

## 셀 3 — 의존성 설치

```python
!pip install -q einops timm pyyaml addict yapf future lpips pytorch-msssim
!pip install -q opencv-python-headless scikit-image tensorboard
print('Deps installed')
```

---

## 셀 4 — 데이터 확인 (이미 Drive에 받아두었다고 가정)

```python
import os
TRAIN_LQ = f'{DATA}/train_patches/input'
TRAIN_GT = f'{DATA}/train_patches/gt'
VAL_LQ   = f'{DATA}/val_patches/input'
VAL_GT   = f'{DATA}/val_patches/gt'

for p in [TRAIN_LQ, TRAIN_GT, VAL_LQ, VAL_GT]:
    if os.path.isdir(p):
        n = len(os.listdir(p))
        print(f'OK  {p}: {n} files')
    else:
        print(f'MISSING  {p}')
```

> ⚠️ 폴더 이름은 SIDL 데이터셋 실제 구조에 맞춰 조정. 보통 `train/input`, `train/gt` 또는 `train_patches/input`, `train_patches/gt` 형태.

---

## 셀 5 — 샘플 시각화 (데이터 무결성 확인)

```python
import matplotlib.pyplot as plt
from PIL import Image
import os, random

names = sorted(os.listdir(TRAIN_LQ))[:5]
fig, axes = plt.subplots(len(names), 2, figsize=(8, 3*len(names)))
for i, name in enumerate(names):
    lq = Image.open(os.path.join(TRAIN_LQ, name))
    gt = Image.open(os.path.join(TRAIN_GT, name))
    axes[i,0].imshow(lq); axes[i,0].set_title(f'LQ: {name}'); axes[i,0].axis('off')
    axes[i,1].imshow(gt); axes[i,1].set_title('GT'); axes[i,1].axis('off')
plt.tight_layout(); plt.show()
```

오염된 입력(LQ)와 깨끗한 GT가 한 쌍으로 보이면 OK.

---

## 셀 6 — NAFNet repo clone

```python
%cd /content
if not os.path.isdir('NAFNet'):
    !git clone https://github.com/megvii-research/NAFNet.git
%cd NAFNet
!pip install -q -r requirements.txt
!python setup.py develop --no_cuda_ext > /dev/null 2>&1
print('NAFNet ready')
```

---

## 셀 7 — Config 파일 복사 (Drive에 미리 둔 yml)

```python
import shutil, os
src = f'{CODE}/nafnet_config_sidl.yml'  # Drive에 미리 업로드 필요
dst = '/content/NAFNet/options/train/SIDL.yml'
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy(src, dst)
print('Config copied:', dst)
```

> ⚠️ Drive의 `SIDL/code/`에 **04_학습_코드/nafnet_config_sidl.yml** 파일을 먼저 업로드해야 함. 그 yml 안의 데이터 경로가 Colab의 `/content/drive/MyDrive/SIDL/data/...`로 되어있는지 확인.

---

## 셀 8 — Sanity Check (1000 iter만 빠르게)

config의 `total_iter: 200000`을 일시적으로 `1000`으로 바꿔서:

```python
import yaml
with open(dst) as f: cfg = yaml.safe_load(f)
cfg['train']['total_iter'] = 1000
cfg['logger']['save_checkpoint_freq'] = 500
cfg['val']['val_freq'] = 500
with open(dst, 'w') as f: yaml.safe_dump(cfg, f)

%cd /content/NAFNet
!python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 \
        basicsr/train.py -opt options/train/SIDL.yml --launcher pytorch
```

기대: loss가 감소하면 OK. 안 떨어지면 데이터 경로 문제일 가능성이 높음.

---

## 셀 9 — 본격 학습 (Sanity check 통과 후)

```python
# total_iter를 다시 200000으로
import yaml
with open(dst) as f: cfg = yaml.safe_load(f)
cfg['train']['total_iter'] = 200000
cfg['logger']['save_checkpoint_freq'] = 5000
cfg['val']['val_freq'] = 5000
with open(dst, 'w') as f: yaml.safe_dump(cfg, f)

%cd /content/NAFNet
!python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 \
        basicsr/train.py -opt options/train/SIDL.yml --launcher pytorch
```

이 셀은 **수 시간~수십 시간** 걸림. 세션 끊겨도 체크포인트는 Drive에 남아있음.

---

## 셀 10 — Resume (세션 끊긴 후 재시작 시)

마지막 `.state` 파일 경로를 찾아서 config에 넣고 다시 실행:

```python
import glob
states = sorted(glob.glob(f'{CKPT}/*/training_states/*.state'))
if states:
    last = states[-1]
    print('Resume from:', last)
    cfg['path'] = cfg.get('path', {})
    cfg['path']['resume_state'] = last
    with open(dst, 'w') as f: yaml.safe_dump(cfg, f)

# 그 다음 셀 9를 다시 실행
```

---

## 셀 11 — Validation (학습 도중에도 가능)

```python
import torch
from PIL import Image
import numpy as np

# 모델 로드
import sys; sys.path.insert(0, '/content/NAFNet')
from basicsr.archs.NAFNet_arch import NAFNet

model = NAFNet(width=32, enc_blk_nums=[2,2,4,8],
               middle_blk_num=12, dec_blk_nums=[2,2,2,2]).cuda()
ckpt = torch.load(f'{CKPT}/.../models/net_g_latest.pth')['params']
model.load_state_dict(ckpt)
model.eval()

# val 이미지 한 장 inference
img = np.array(Image.open(os.path.join(VAL_LQ, sorted(os.listdir(VAL_LQ))[0]))) / 255.0
x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()
with torch.no_grad():
    y = model(x)
y_np = (y.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)

plt.subplot(1,2,1); plt.imshow(img); plt.title('Input')
plt.subplot(1,2,2); plt.imshow(y_np); plt.title('Output')
plt.show()
```

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| Loss가 NaN | LR이 너무 높음 → 1e-4로 낮춰서 재시작 |
| OOM | batch_size를 4 또는 2로, gt_size를 192로 |
| 학습 안 시작됨 | 데이터 경로 확인. yml의 `dataroot_lq`, `dataroot_gt` 절대 경로? |
| Colab 끊김 | Drive에 ckpt 있는지 확인 → resume |
| `cudaErrorIllegalAddress` | T4 + PyTorch 호환 문제. PyTorch 2.1로 다운그레이드 시도 |
| Validation PSNR 안 뜸 | val 데이터셋 폴더에 입력/GT 모두 있는지 확인 |
