# Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models
The official implementation of [*Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models*](https://arxiv.org/pdf/2505.15130).
![AdvCLIP-LoRA](AdvCLIP-LoRA.pdf "AdvCLIP-LoRA-pipeline")
## How to Run

You can run `main.py` with some specified arguments.

## Data Preparation
Please follow the instructions at CoOP https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md to prepare the datasets.

### Training

`--root_path` takes as input a path to a dataset.

`--backbone` name of the backbone model.

`--epsilon_train` epsilon used for training.

`--epsilon` epsilon used for PGD or FGSM attacks.

You can optionally provide a save_path to save the LoRA modules, which can be reloaded easily with the --eval_only argument. 

### Running example

Clean and Robust Few-shot: `bash scripts/few-shot.sh`


## Acknowledgement

We would like to thank the authors for releasing the public repository: [CLIP-LoRA](https://github.com/aheldis/CLIP-LoRA).

## Citation
If you find this project helpful, please consider citing the following paper:
```
@article{ghiasvand2025few,
  title={Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models},
  author={Ghiasvand, Sajjad and Oskouie, Haniyeh Ehsani and Alizadeh, Mahnoosh and Pedarsani, Ramtin},
  journal={arXiv preprint arXiv:2505.15130},
  year={2025}
}
```
