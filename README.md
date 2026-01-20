# Adaptive Cross-Task Learning for Speech Emotion Recognition: A Progressive Multi-Task Framework with Structured Unfreezing

This project is the official open-source implementation of the paper *“Adaptive Cross-Task Learning for Speech Emotion Recognition: A Progressive Multi-Task Framework with Structured Unfreezing”*, primarily intended to reproduce and validate the methods and experimental results presented in the paper.

**Authors**: Zhuorui Li, Xianhong Chen, and Maoshen Jia

🛠 Requirements  
`pytorch==2.2.1`  
`python==3.8.18`  
`yaml==6.0.2`  
`numpy==1.24.4`  
`scikit-learn==1.3.2`

📂 Datasets  
[IEMOCAP](https://sail.usc.edu/iemocap/)

📄 Code Structure  
IEMOCAP Experiments  
The `IEMOCAP_exp` directory contains experiments on the IEMOCAP dataset, where 5,531 utterances are used for 4-class emotion classification (anger, happiness + excited, neutral, sadness). A 5-fold cross-validation strategy is used for training.  

`config/config.yml`: Configuration file for the experiments.  
`main.py`: Main program for running experiments.  
`utils.py`: Various utility methods used throughout.  
`model.py`: File containing the model definition.  
`output/`: Directory for storing experimental result logs.
