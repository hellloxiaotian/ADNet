# ADNet
## Atention-guided CNN for image denoising(Neural Networks,2020)

## Absract
### Deep convolutional neural networks (CNNs) have attracted considerable interest in low-level computer vision. Researches are usually devoted to improving the performance via very deep CNNs. However, as the depth increases, influences of the shallow layers on deep layers are weakened. Inspired by the fact, we propose an attention-guided denoising convolutional neural network (ADNet), mainly including a sparse block (SB), a feature enhancement block (FEB), an attention block (AB) and a reconstruction block (RB) for image denoising. Specifically, the SB makes a tradeoff between performance and efficiency by using dilated and common convolutions to remove the noise. The FEB integrates global and local features information via a long path to enhance the expressive ability of the denoising model. The AB is used to finely extract the noise information hidden in the complex background, which is very effective for complex noisy images, especially real noisy images and bind denoising. Also, the FEB is integrated with the AB to improve the efficiency and reduce the complexity for training a denoising model. Finally, a RB aims to construct the clean image through the obtained noise mapping and the given noisy image. Additionally, comprehensive experiments show that the proposed ADNet performs very well in three tasks (i.e. synthetic and real noisy images, and blind denoising) in terms of both quantitative and qualitative evaluations. The code of ADNet is accessible at https://github.com/hellloxiaotian/ADNet.

## Requirements (Pytorch)
### Pytorch 0.41
### Python 2.7



## Commands
## Training
### Training datasets 
#### The  training dataset of the gray noisy images is downloaded at https://pan.baidu.com/s/13jDZfayiM-KxLAOVWvFlUA
#### The  training dataset of the color noisy images is downloaded at https://pan.baidu.com/s/1cx3ymsWLIT-YIiJRBza24Q

### Train ADNet-S (ADNet with known noise level)
#### python train.py --prepropcess True --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25

### Train ADNet-B (DnCNN with blind noise level)
#### python train.py --preprocess True --num_of_layers 17 --mode B --val_noiseL 25

## Test 
### Gray noisy images
#### python test.py --num_of_layers 17 --logdir g15 --test_data Set68 --test_noiseL 15 
### Gray blind denoising
#### python test_Gb.py --num_of_layers 17 --logdir gblind --test_data Set68 --test_noiseL 25 

### Color noisy images
#### python test_c.py --num_of_layers 17 --logdir g15 --test_data Set68 --test_noiseL 15 
### Color blind denoising
#### python test_c.py --num_of_layers 17 --logdir cblind --test_data Set68 --test_noiseL 15


## Test Results
### ADNet for BSD68

