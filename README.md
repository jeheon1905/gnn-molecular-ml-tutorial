# GNN Molecular ML Tutorial

Graph Neural Networks (GNN)을 활용한 분자 특성 예측 학습 자료입니다.

## 📚 Contents

- **notebooks/**: Jupyter 노트북 튜토리얼
  - `rdkit_gnn_preprocessing_tutorial.ipynb`: RDKit을 활용한 분자 그래프 전처리
  - `freesolv_finetune_tutorial.ipynb`: GCN 모델 Fine-tuning (예정)

- **external/MolCLR**: 분자 표현 학습 프레임워크 (Git submodule)

## 🚀 Setup

### 1. Clone Repository

```bash
git clone --recurse-submodules https://github.com/<your-username>/gnn-molecular-ml-tutorial.git
cd gnn-molecular-ml-tutorial
```

이미 클론한 경우:
```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

```bash
conda create -n molclr python=3.8
conda activate molclr

# PyTorch (CUDA 버전에 맞게 수정)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# PyTorch Geometric
pip install torch-geometric

# RDKit
conda install -c conda-forge rdkit

# Other dependencies
pip install pandas matplotlib scikit-learn pyyaml gdown
```

### 3. Download Datasets

```bash
# gdown 설치 (이미 설치되어 있으면 skip)
pip install gdown

# MolCLR 데이터셋 다운로드
gdown "https://drive.google.com/uc?id=1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE"
unzip molclr_data.zip
```

## 📖 Tutorials

### 1. RDKit GNN Preprocessing
분자 SMILES를 GNN 입력 데이터로 변환하는 전처리 과정을 학습합니다.

```bash
jupyter notebook notebooks/rdkit_gnn_preprocessing_tutorial.ipynb
```

**학습 내용:**
- SMILES 표기법 이해
- RDKit Mol 객체 생성 및 조작
- 원자(Atom) 특징 추출
- 결합(Bond) 특징 추출
- 분자 그래프 구조 (인접 행렬, 엣지 리스트)
- PyTorch Geometric Data 객체 생성

### 2. FreeSolv Fine-tuning (예정)
사전 학습된 GCN 모델을 FreeSolv 데이터셋에 fine-tuning합니다.

## 📁 Project Structure

```
gnn-molecular-ml-tutorial/
├── .gitmodules              # Git 서브모듈 설정
├── external/
│   └── MolCLR/             # MolCLR 프레임워크 (submodule)
├── notebooks/
│   └── rdkit_gnn_preprocessing_tutorial.ipynb
├── data/                    # 데이터셋 (다운로드 후 생성)
└── README.md
```

## 🔗 References

- [MolCLR](https://github.com/yuyangw/MolCLR): Molecular Contrastive Learning of Representations
- [RDKit](https://www.rdkit.org/): Open-source cheminformatics toolkit
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): Graph neural network library

## 📝 License

튜토리얼 코드는 MIT License를 따릅니다.
MolCLR 코드는 원저작자의 라이센스를 따릅니다.
