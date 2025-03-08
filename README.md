# FedVSR: Federated Learning for Video Super-Resolution

This repository contains code and experiments for **FedVSR**, a method to train Video Super-Resolution (VSR) models under the Federated Learning (FL) paradigm. The goal is to enable high-quality VSR while ensuring data privacy by not centralizing raw video data. This README provides instructions for installing dependencies, running experiments, and reproducing results.

---

## Project Overview

- **What is FedVSR?**  
  FedVSR is a federated training framework that allows multiple clients to train a shared video super-resolution model without sharing their raw data. This approach preserves data privacy and is suitable for scenarios where raw video footage is sensitive.

- **Key Features**  
  - Model-agnostic: Works with various VSR architectures.  
  - Reproducible Experiments: Scripts to reproduce training, testing, and generate performance figures (e.g., PSNR, SSIM).  
  - Scalable Data Handling: Supports different datasets and automatically handles training/evaluation splits.

<p align="center">
  <img width="800" src="sample.png">
</p>

> *Note: This code is released under review status; therefore, no direct citations are referenced in this repo.*

---

## Repository Structure

```
.
├── Kinetics_Scripts/
│   └── copy_movie.py          # Script for extracting & copying related movies
│
├── VRTandRVRT/
│   ├── vrt_and_rvrt_{ALG}.py  # Scripts to run all trainings with different methods.     
│   ├── main_test_{MODEL}      # Scripts to run tests on different models    
│   └── 
│
├── IART/
│   ├── iart_{ALG}.py          # Scripts to run all trainings with different methods.        
│   └── test_scripts/test_*.py # Scripts to run tests  
│
├── requirements.txt      # Python package requirements
└── README.md             # Main documentation
```

---

## Installation & Setup

1. **Clone the repository**  

2. **Set up Python environment**  
   - **Using `requirements.txt`:**  
     ```bash
     pip install -r requirements.txt
     ```

## Prepare Data:

To prepare the dataset, follow [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#Video-Super-Resolution). After completing the preparation, the directory structure should be as follows: 

```
datasets/
├──REDS/
│   └──val_REDS4_sharp
│   └──val_REDS4_sharp_bicubic
```

## Training

For VRT and RVRT prepare the related json in the file and the run the scripts:
```bash
# VSR trained on FedAvg
python VRTandRVRT/vrt_and_rvrt_fedavg.py
# VSR trained on FedVSR
python VRTandRVRT/vrt_and_rvrt_fedvsr.py
```

For IART:
```bash
# VSR trained on FedAvg
python IART/iart_fedavg.py
# VSR trained on FedVSR
python IART/iart_fedvsr.py
```

## Testing

You can test using 2 different datasets VID4 and REDS:
```bash
python VRTandRVRT/main_test_rvrt.py --task 001_RVRT_videosr_bi_REDS_30frames
python VRTandRVRT/main_test_vrt.py --task 001_RVRT_videosr_bi_REDS_30frames
```

To test IART you would use:

```bash
python IART/test_scripts/test_IART_REDS4_N6.py 
python IART/test_scripts/test_IART_Vid4_N6.py
```

## Results

<p align="center">
  <img width="800" src="result_graph.png">
</p>

<p align="center">
  <img width="800" src="result_table.png">
</p>


## License

```
MIT License
```
---
