# The Official Implementation of Intervention-Driven Correlation Reduction (ICDE 2025)

## Usage
Law School Dataset
```bash
python idcr_train.py -l 0.85 --epochs 400 --batch_size 2048 --data_name law_school
```
Synthetic Dataset
```bash
python idcr_train.py -l 0.8 --epochs 1000 --batch_size 2048 --data_name synthetic
```
## Environment
* python 3.6
* torch 1.10.1+cuda111
* scikit-learn 0.24.2
* numpy 1.19.5
* pandas 1.1.5
## Cite
```bibtex
The paper is not yet available, please follow the proceedings of ICDE 2025
```
This project is based on modifications of the Causal-TGAN code. If you use this code, please cite the original Causal-TGAN paper as follows:
```bibtex
@inproceedings{
    wen2022causaltgan,
    title={Causal-{TGAN}: Modeling Tabular Data Using Causally-Aware {GAN}},
    author={Bingyang Wen and Yupeng Cao and Fan Yang and Koduvayur Subbalakshmi and Rajarathnam Chandramouli},
    booktitle={ICLR Workshop on Deep Generative Models for Highly Structured Data},
    year={2022},
    url={https://openreview.net/forum?id=BEhxCh4dvW5}
}
```