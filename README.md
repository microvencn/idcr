# The Official Implementation of Intervention-Driven Correlation Reduction (ICDE 2025)
**Paper Title.** Intervention-Driven Correlation Reduction: A Data Generation Approach for Achieving Counterfactually Fair Predictors

**Abstract.** Achieving counterfactual fairness is a critical objective in advancing fairness research within machine learning. Studies have shown that machine learning models often inherit biases from their training data, leading to unfair decision-making. Fair data generation methods aim to mitigate these biases, ensuring that predictors trained on such data uphold fairness. However, in the context of counterfactual fairness, existing methods for generating fair data are often limited in their applicability and lead to significant performance losses in downstream predictors. To address these issues, this paper proposes a new algorithm for generating counterfactually fair data, allowing predictors trained on this generated data to adhere to counterfactual fairness. We propose a new metric, Intervention-Driven Correlation (IDC), to evaluate the counterfactual fairness of generative models. IDC assesses fairness by applying random interventions to samples and measuring the statistical correlation between the degree of intervention and the outcome of interest. This metric is applicable to both discrete and continuous sensitive attributes and labels. Furthermore, our studies reveal a critical insight: counterfactually fair data does not always guarantee counterfactually fair predictors when deployed in real-world scenarios. We identify the root causes of this issue and propose a robust solution. To bridge this gap, we propose the IDC-Reduction method, which ensures the fairness of downstream predictors by generating counterfactually fair data. Experimentally, our method outperforms existing approaches and achieves counterfactual fairness regardless of the type of downstream predictors.

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
@inproceedings{zhou2025intervention,
    author={Zhou, Dehua and Wu, Bowei and Wang, Ke  and Yang, Qifen and Yuhui, Deng and Siu Ming, Yiu},
    booktitle={41st {IEEE} International Conference on Data Engineering, {ICDE} 2025}, 
    title={Intervention-Driven Correlation Reduction: A Data Generation Approach for Achieving Counterfactually Fair Predictors}, 
    year={2025},
}
```
This project is based on modifications of the Causal-TGAN code. If you use this code, please cite the original Causal-TGAN paper as follows:
```bibtex
@inproceedings{wen2022causaltgan,
    title={Causal-{TGAN}: Modeling Tabular Data Using Causally-Aware {GAN}},
    author={Bingyang Wen and Yupeng Cao and Fan Yang and Koduvayur Subbalakshmi and Rajarathnam Chandramouli},
    booktitle={ICLR Workshop on Deep Generative Models for Highly Structured Data},
    year={2022},
    url={https://openreview.net/forum?id=BEhxCh4dvW5}
}
```