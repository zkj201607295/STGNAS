TSGNAS: Topology and Semantic Graph Neural Architecture Search

1. Environment Setup
   
```bash
torch == 1.12.1
torch-geometric == 2.3.1
numpy
scipy
tqdm
```

2. Dataset Preparation

Place your graph datasets under the data/ directory.
Each dataset folder should include at least:
```bash
data/
 ├── CiteSeer/
 ├── PubMed/
 ├── Photo/
 ├── Actor/
 └── ...
```

3. Run the Main Program

The main script can be executed directly:
```bash
python main.py --dataset Photo
```

