# Federated-Learning-Framework
This is partly the reproduction of the paper of [Privacy-Preserving Federated Learning in Fog Computing](DOI: 10.1109/JIOT.2020.2987958. 2020)   
Only experiments on MNIST  is produced by far.

## Requirements
python>=3.6  
pytorch>=0.4

## Set up
run blind_server.py and model_server.py，Choose the appropriate cilent operation according to the number of participants in the mod

## Abstract
Federated learning can combine a large number of scattered user groups and train models collaboratively without uploading data sets, so as to avoid the server collecting user sensitive data. However, the model of federated learning will expose the training set information of users, and the uneven amount of data owned by users in multiple users’ scenarios will lead to the inefficiency of training. In this article, we propose a privacy-preserving federated learning scheme in fog computing. Acting as a participant, each fog node is enabled to collect Internet-of-Things (IoT) device data and complete the learning task in our scheme. Such design effectively improves the low training efficiency and model accuracy caused by the uneven distribution of data and the large gap of computing power. We enable IoT device data to satisfy $\varepsilon $ -differential privacy to resist data attacks and leverage the combination of blinding and Paillier homomorphic encryption against model attacks, which realize the security aggregation of model parameters. In addition, we formally verified our scheme can not only guarantee both data security and model security but completely resist collusion attacks launched by multiple malicious entities. Our experiments based on the Fashion-MNIST data set prove that our scheme is highly efficient in practice.

