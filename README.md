# GNN Molecular ML Tutorial

Graph Neural Networks (GNN)을 활용한 분자 특성 예측 학습 자료입니다.

## 📚 Contents

- **notebooks/**: Jupyter 노트북 기반 실습 자료
  1. `rdkit_molecular_graph_tutorial.ipynb`
     → SMILES부터 RDKit Mol 객체 및 분자 그래프 구성
  2. `gcn_basics_tutorial.ipynb`
     → GCN 기본 수식, 메시지 패싱, forward propagation 이해
  3. `freesolv_finetune_tutorial.ipynb`
     → FreeSolv 데이터셋에서 GCN fine-tuning
       → Pretraining 유무에 따른 학습 곡선 비교

- **external/MolCLR**
  Self-supervised molecular representation learning framework
  (Git submodule)

---

## 🚀 Setup

### 1. Clone Repository

```bash
git clone https://github.com/jeheon1905/gnn-molecular-ml-tutorial.git
cd gnn-molecular-ml-tutorial

git submodule update --init --recursive  # install MolCLR
```

### 2. Environment Setting

#### Create Conda Environment

```bash
# conda environment 생성
conda create -y -n gnn-tutorial python=3.10

# 환경 활성화
conda activate gnn-tutorial
```

#### Install PyTorch (CUDA 11.8)

```bash
# PyTorch with CUDA 11.8 support
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# NumPy (2.x 충돌 방지)
pip install numpy==1.26.4
```

#### Install PyTorch Geometric

```bash
# PyG core
pip install torch-geometric

# PyG CUDA extensions (torch 2.2.2 + cu118)
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
```

#### Install Other Dependencies

```bash
# RDKit
conda install -y -c conda-forge rdkit

# Visualization and ML tools
conda install -y -c conda-forge seaborn
pip install scikit-learn==1.4.2 pandas matplotlib
```

#### Install Jupyter Kernel

```bash
# Jupyter kernel 등록
pip install ipykernel
python -m ipykernel install --user --name gnn-tutorial --display-name "Python (gnn-tutorial)"
```

### 3. Sanity Checks

#### PyTorch / CUDA Check

```bash
python << 'EOF'
import numpy as np
import torch

print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
EOF
```

**정상 출력 예:**
```
NumPy: 1.26.4
Torch: 2.2.2+cu118
CUDA: 11.8
CUDA available: True
```

#### PyTorch Geometric Check

```bash
python << 'EOF'
import torch
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(4, 16).to(device)
edge_index = torch.tensor([[0,1,2],
                           [1,2,3]]).to(device)

conv = GCNConv(16, 32).to(device)
out = conv(x, edge_index)

print("PyG OK:", out.shape)
EOF
```

**정상 출력:**
```
PyG OK: torch.Size([4, 32])
```

#### RDKit Check

```bash
python << 'EOF'
from rdkit import Chem
from rdkit.Chem import Draw
m = Chem.MolFromSmiles("CCO")
print("RDKit OK:", m)
EOF
```

**정상 출력:**
```
RDKit OK: <rdkit.Chem.rdchem.Mol object at 0x...>
```

## 📖 Tutorials

### 1. RDKit Molecular Graph Tutorial
분자 SMILES를 GNN 입력 데이터로 변환하는 전처리 과정을 학습합니다.

```bash
jupyter lab notebooks/rdkit_molecular_graph_tutorial.ipynb
```

**학습 내용:**
- SMILES 표기법 이해
- RDKit Mol 객체 생성 및 조작
- 원자(Atom) 특징 추출
- 결합(Bond) 특징 추출
- 분자 그래프 구조 (인접 행렬, 엣지 리스트)
- PyTorch Geometric Data 객체 생성

### 2. GCN Basics Tutorial
GCN(Graph Convolutional Network)의 작동 원리를 단계별로 이해합니다.

```bash
jupyter lab notebooks/gcn_basics_tutorial.ipynb
```

**학습 내용:**
- GCN의 수학적 정의 및 구현
- GCN Layer 단계별 분석
- 다층 GCN 구조
- Graph Pooling 방법
- 완전한 GCN 모델 구현
- Node Permutation 불변성 (Permutation Invariance)

### 3. FreeSolv Fine-tuning Tutorial
사전 학습된 GCN 모델을 FreeSolv 데이터셋에 fine-tuning하고 pretraining 효과를 비교합니다.

```bash
jupyter lab notebooks/freesolv_finetune_tutorial.ipynb
```

**학습 내용:**
- FreeSolv 데이터셋 탐색
- MolCLR 데이터 로더 사용
- Random initialization vs Pre-trained 모델 비교
- Transfer Learning 효과 분석
- 학습 곡선 및 오차 분석

## 📁 Project Structure

```
gnn-molecular-ml-tutorial/
├── .gitmodules                     # Git 서브모듈 설정
├── external/
│   └── MolCLR/                    # MolCLR 프레임워크 (submodule)
├── notebooks/
│   ├── rdkit_molecular_graph_tutorial.ipynb
│   ├── gcn_basics_tutorial.ipynb
│   └── freesolv_finetune_tutorial.ipynb
├── data/
│   └── freesolv/                  # FreeSolv 데이터셋
└── README.md
```

## 🔗 References

- [MolCLR](https://github.com/yuyangw/MolCLR): Molecular Contrastive Learning of Representations
- [RDKit](https://www.rdkit.org/): Open-source cheminformatics toolkit
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): Graph neural network library

## 📝 License

- 튜토리얼 코드는 MIT License를 따릅니다.
- MolCLR 코드는 원저작자의 라이센스를 따릅니다.
