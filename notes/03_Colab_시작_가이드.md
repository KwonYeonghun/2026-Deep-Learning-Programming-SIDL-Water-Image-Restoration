# 03. Colab으로 시작하는 가이드

## 결론부터 — 지금 Colab으로 시작해도 되는가?

✅ **가능합니다, 단 patch 버전 데이터(512×512)만 쓰는 한.**

### Colab의 한계와 우리 대응

| 한계 | 우리의 대응 |
|---|---|
| GPU: T4 16GB (무료) / V100·L4 (Pro) | NAFNet width-32 / Restormer-light 정도면 OK |
| 세션 12시간 자동 끊김 (무료) | 5K iter마다 Drive에 체크포인트 → resume 가능 |
| RAM 12.7GB | patch 데이터 = OK / full-res = 안 됨 |
| 디스크 ~78GB (런타임 공간) | 데이터를 Drive에 두고 mount |
| 백그라운드 학습 불가 | 브라우저 탭 살려두기, keep-alive 스크립트 |

### 데이터 크기 예상

- **patch 버전 (512×512 PNG)**: train ~1500장 × 2(input/gt) ≈ **8~12GB** → Drive에 충분
- **full-res 4032×3024**: 30~50GB → Colab은 비추, 데스크탑에서만
- **DNG RAW**: 60~100GB → 데스크탑 전용

→ **Colab은 patch만**, full-res inference는 나중에 데스크탑(5060 Ti)에서.

---

## ⚙️ Colab 환경 세팅 (최초 1회)

### 1. 노트북 만들기
1. Google Drive → 새로 만들기 → 더보기 → Google Colaboratory
2. 노트북 이름: `SIDL_NAFNet_Train`
3. 런타임 → 런타임 유형 변경 → **GPU (T4)** (무료) 또는 **V100/A100** (Pro)

### 2. Drive mount
```python
from google.colab import drive
drive.mount('/content/drive')

# 작업 폴더 생성
import os
WORK = '/content/drive/MyDrive/SIDL'
os.makedirs(WORK, exist_ok=True)
os.makedirs(f'{WORK}/data', exist_ok=True)
os.makedirs(f'{WORK}/checkpoints', exist_ok=True)
os.makedirs(f'{WORK}/code', exist_ok=True)
print('OK:', os.listdir(WORK))
```

### 3. GPU 확인
```python
import torch
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
# 기대 출력: CUDA: True / Device: Tesla T4 / VRAM: ~16 GB
```

### 4. 의존성 설치
```python
!pip install -q einops timm pyyaml addict yapf future lpips pytorch-msssim tensorboard
!pip install -q opencv-python-headless scikit-image
```

---

## 📦 데이터셋 다운로드 (Day 2 — 핵심)

### 방법 1: Drive에 직접 받아두기 (가장 추천)

브라우저로 SIDL 프로젝트 페이지에서 데이터 링크들을 클릭 → "내 드라이브에 추가" 또는 "사본 만들기"로 복사하면, 아래 mount된 경로에서 직접 접근 가능. **다운로드 트래픽도 안 잡아먹음**.

### 방법 2: Colab 셀에서 gdown으로
```python
%cd /content/drive/MyDrive/SIDL/data
!pip install -q gdown

# train patches (폴더)
!gdown --folder https://drive.google.com/drive/folders/1Bmw2s0OfnwsAEdd81er_80dP4m9IlvgO

# val patches (단일 파일)
!gdown 13sClVSqtWCZNiApykjtDfFCFe35OtauR
!unzip -q val_patches.zip -d val_patches  # 파일명 실제 받은 것에 맞게

# test patches
!gdown 1exvxODE1w4AitIqkRmB_XqCow3YlzLYj
!unzip -q test_patches.zip -d test_patches
```

### 받은 후 폴더 구조 점검
```python
import os
ROOT = '/content/drive/MyDrive/SIDL/data'
for sub in os.listdir(ROOT):
    sub_path = os.path.join(ROOT, sub)
    if os.path.isdir(sub_path):
        n = len([f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg'))])
        print(f'{sub}: {n} files')

# 기대 형태 (실제 SIDL 구조에 따라 조정):
# train_patches/input: 1500
# train_patches/gt:    1500
# val_patches/input:   ...
# val_patches/gt:      ...
```

⚠️ 실제 다운받은 zip 안의 폴더 이름이 다를 수 있으니 확인 후 코드의 경로를 맞춰주세요.

---

## 🔥 학습 코드 시작 — 두 갈래

### A. 직접 NAFNet repo 사용 (권장 — 검증된 코드)
```python
%cd /content
!git clone https://github.com/megvii-research/NAFNet.git
%cd NAFNet
!pip install -q -r requirements.txt
!python setup.py develop --no_cuda_ext

# 우리 config 복사
import shutil
shutil.copy('/content/drive/MyDrive/SIDL/code/nafnet_config_sidl.yml',
            'options/train/SIDL.yml')

# 학습 시작
!python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 \
        basicsr/train.py -opt options/train/SIDL.yml --launcher pytorch
```

### B. 단순 자체 구현 (학습용, 04_학습_코드 폴더 참고)

처음 익히려면 우리가 만든 단순 학습 스크립트로 시작하는 것도 좋음. 04_학습_코드/colab_train_nafnet.ipynb_template.md 참조.

---

## 🛡️ 세션 끊김 대비 (무료 Colab 필수)

### 1. 체크포인트는 항상 Drive에 저장
```yaml
# config 안에서
path:
  experiments_root: /content/drive/MyDrive/SIDL/checkpoints
```

### 2. Resume이 가능하도록 항상 latest checkpoint 갱신
NAFNet/BasicSR은 `--auto_resume` 옵션 또는 `path.resume_state` 설정으로 자동 resume 됨.

```yaml
path:
  resume_state: ~  # 최초 학습
  # resume 시: /content/drive/MyDrive/SIDL/checkpoints/.../training_states/xxxx.state
```

### 3. Keep-alive 스크립트 (브라우저 console에 붙여넣기)
```javascript
function ClickConnect() {
  document.querySelector("colab-connect-button")?.click();
  console.log("clicked", new Date());
}
setInterval(ClickConnect, 60 * 1000);
```

### 4. 끊긴 후 재시작 절차
1. Colab 새 세션 열고 Drive mount
2. NAFNet repo 다시 clone (또는 Drive에 미리 복사해둔 버전)
3. config의 `resume_state`에 마지막 .state 파일 경로 지정
4. 학습 명령 다시 실행

---

## 💰 Colab 무료 vs Pro 비교

| | 무료 | Pro (\$10/월) | Pro+ (\$50/월) |
|---|---|---|---|
| GPU | T4 16GB | V100/L4/A100 (선택 가능) | A100 우선 |
| 세션 시간 | 약 12시간 | 24시간 | 24시간 + 백그라운드 |
| RAM | 12.7GB | 25-51GB | 51GB |
| 학습 속도 (T4 vs V100) | 1.0x | ~2.5x | ~3-5x (A100) |
| 우리 프로젝트 적합도 | 가능, 단 시간 더 걸림 | **추천** | 시간 충분 |

**한 달 프로젝트라 Pro 1개월 결제(\$10) 권장**. 무료로도 가능하지만 시간이 빠듯할 수 있음.

---

## 🔄 Colab에서 데스크탑(5060 Ti)으로 전환은 언제?

| 상황 | Colab | 데스크탑 |
|---|---|---|
| 환경 셋업 / pipeline 검증 | ✅ | ⚪ |
| 베이스라인 학습 | ✅ | ✅ |
| Long training (200K+ iter) | △ Pro면 OK | ✅ |
| Full-res inference | ❌ | ✅ |
| 여러 모델 동시 학습 | ❌ | ✅ |
| Ensemble 실험 | ✅ | ✅ |
| **권장 분담** | Day 1-7 (Week 1) | Week 2부터 메인 |

→ **첫 주는 Colab으로 빠르게 검증, 둘째 주부터 데스크탑이 메인이고 Colab은 보조 실험용**.

---

## 🎯 오늘 (Day 1) Colab에서 진짜 해야 할 것

```text
1) Drive 폴더 만들기 (위 셋업 코드)
2) GPU 잡혔는지 확인
3) PyTorch CUDA 동작 확인
4) (시간 되면) 데이터 다운로드 시작 — 백그라운드로 둘 수 있음
```

오늘은 여기까지만. 데이터 다운로드는 시간 걸리므로 자기 전에 걸어두면 됨.
