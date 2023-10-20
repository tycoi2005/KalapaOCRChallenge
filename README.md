# Huynh Duc Toan - Deep Learning and Reinforcement Learning week 9 Project

I used data from [Kalapa OCR challengen 2023](https://challenge.kalapa.vn/portal/handwritten-vietnamese-text-recognition/overview) and developed an model to do text recognition for hanwritten vietnamese.

## Data describsion

### Files and Folders

Training data

```
training_data
|___ images
     |___ 001
		      |___ 1.jpg
					|___ ...
          |___ 26.jpg
		 |___ ...
     |___ 150
          |___ 1.jpg
					|___ ...
          |___ 26.jpg
|___ annotations
     |___ 001.txt
     |___ 002.txt
     |___ ...
     |___ 150.txt
```

**images** folder contains subfolders are id of writter, each subfolder contains many images, each image is one line handwriting address in vietnamese

**annotations** folder contain text files that have names are ids of writter, each file contain labels of images. imagepaths and labels are seperated by *tab* character ("\\t")
```
7/0.jpg	Ái Quốc Tp Hải Dương Hải Dương
7/1.jpg	Tổ 5 Ấp Suối Đục Sông Nhạn Cẩm Mỹ Đồng Nai
```

Raw images look like this:
![](./images/02ImagesRaw.png)

```python
# preprocess these images:
def preprocessimage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    new_height = 163
    new_width = int((163/img.shape[0])*img.shape[1])
    img = cv2.resize(img, (new_width, new_height))
    if (img.shape[1]<max_upwidth):
        # pad right image
        img = np.pad(img, ((0,0),(0, max_upwidth-new_width)), 'median')
    else:
        # crop right image
        print(img.shape)
        img = img[0:new_height, 0:max_upwidth]
        print(img.shape)
    # Blur image:
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Threshold the image using adapative threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return img
```

![](./images/01Preprocess.png)

## Main objects of this analysis

- Create a lightweight (size < 50 MB) model that could read handwriting vietnamese effectively

- Model inference < 2s

- Model could work offline without the internet

## Variations of Deep Learning Models for Handwritten Text Recognition

Handwritten Text Recognition (HTR) is a challenging task in the field of machine learning and artificial intelligence due to the high variability of handwriting styles and the complexity of languages. Several deep learning models have been proposed to tackle this problem. Here are a few variations that could be considered for this analysis:

- Convolutional Neural Networks (CNNs): CNNs are widely used in image processing tasks, including HTR. They can extract local features from images and are invariant to local transformations. A simple CNN model for HTR might consist of several convolutional layers followed by fully connected layers and a softmax activation function for classification.

- Recurrent Neural Networks (RNNs): RNNs are designed to work with sequence data. In the context of HTR, the input image can be treated as a sequence where each timestep corresponds to a column of pixels in the image. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are two popular types of RNNs used in HTR tasks.

- Connectionist Temporal Classification (CTC): CTC is a popular loss function for sequence-to-sequence problems without needing an alignment between input and output. It’s often used with RNNs in HTR tasks.

- CNN + RNN + CTC: A common architecture for HTR tasks is to use CNN layers to extract features from the input images, followed by RNN layers to process the sequence of features, and finally use CTC for sequence prediction. This architecture combines the strengths of CNNs, RNNs, and CTC.

- Transformers: Transformers, which are based on self-attention mechanisms, have recently shown great success in many NLP tasks. They could also be adapted for HTR tasks.

- CRNN (Convolutional Recurrent Neural Network): This model combines CNN and RNN to extract robust features from handwriting images and predict sequences respectively.

Given the main objectives of this analysis, which are creating a lightweight model that can read handwritten Vietnamese effectively, work offline without internet, and have an inference time less than 2 seconds, I choosed the CRN + RNN + CTC model.


## Model Summary

| Layer (type)                                | Output Shape           | Param # | Connected to                                            |
|---------------------------------------------|------------------------|---------|---------------------------------------------------------|
| input_17 (InputLayer)                       | [(None, 163, 3002, 1)] | 0       | []                                                      |
| conv2d_105 (Conv2D)                         | (None, 163, 3002, 64)  | 640     | ['input_17[0][0]']                                      |
| max_pooling2d_56 (MaxPooling2D)             | (None, 54, 1000, 64)   | 0       | ['conv2d_105[0][0]']                                    |
| activation_100 (Activation)                 | (None, 54, 1000, 64)   | 0       | ['max_pooling2d_56[0][0]']                              |
| conv2d_106 (Conv2D)                         | (None, 54, 1000, 64)   | 36928   | ['activation_100[0][0]']                                |
| max_pooling2d_57 (MaxPooling2D)             | (None, 17, 333, 64)    | 0       | ['conv2d_106[0][0]']                                    |
| activation_101 (Activation)                 | (None, 17, 333, 64)    | 0       | ['max_pooling2d_57[0][0]']                              |
| conv2d_107 (Conv2D)                         | (None, 17, 333, 128)   | 73856   | ['activation_101[0][0]']                                |
| batch_normalization_73 (BatchNormalization) | (None, 17, 333, 128)   | 512     | ['conv2d_107[0][0]']                                    |
| activation_102 (Activation)                 | (None, 17, 333, 128)   | 0       | ['batch_normalization_73[0][0]']                        |
| conv2d_108 (Conv2D)                         | (None, 17, 333, 128)   | 147584  | ['activation_102[0][0]']                                |
| batch_normalization_74 (BatchNormalization) | (None, 17, 333, 128)   | 512     | ['conv2d_108[0][0]']                                    |
| add_31 (Add)                                | (None, 17, 333, 128)   | 0       | ['batch_normalization_74[0][0]','activation_102[0][0]'] |
| activation_103 (Activation)                 | (None, 17, 333, 128)   | 0       | ['add_31[0][0]']                                        |
| conv2d_109 (Conv2D)                         | (None, 17, 333, 256)   | 295168  | ['activation_103[0][0]']                                |
| batch_normalization_75 (BatchNormalization) | (None, 17, 333, 256)   | 1024    | ['conv2d_109[0][0]']                                    |
| activation_104 (Activation)                 | (None, 17, 333, 256)   | 0       | ['batch_normalization_75[0][0]']                        |
| conv2d_110 (Conv2D)                         | (None, 17, 333, 256)   | 590080  | ['activation_104[0][0]']                                |
| batch_normalization_76 (BatchNormalization) | (None, 17, 333, 256)   | 1024    | ['conv2d_110[0][0]']                                    |
| add_32 (Add)                                | (None, 17, 333, 256)   | 0       | ['batch_normalization_76[0][0]','activation_104[0][0]'] |
| activation_105 (Activation)                 | (None, 17, 333, 256)   | 0       | ['add_32[0][0]']                                        |
| conv2d_111 (Conv2D)                         | (None, 17, 333, 512)   | 1180160 | ['activation_105[0][0]']                                |
| batch_normalization_77 (BatchNormalization) | (None, 17, 333, 512)   | 2048    | ['conv2d_111[0][0]']                                    |
| max_pooling2d_58 (MaxPooling2D)             | (None, 5, 333, 512)    | 0       | ['batch_normalization_77[0][0]']                        |
| activation_106 (Activation)                 | (None, 5, 333, 512)    | 0       | ['max_pooling2d_58[0][0]']                              |
| max_pooling2d_59 (MaxPooling2D)             | (None, 1, 333, 512)    | 0       | ['activation_106[0][0]']                                |
| lambda_9 (Lambda)                           | (None, 333, 512)       | 0       | ['max_pooling2d_59[0][0]']                              |
| bidirectional_14 (Bidirectional)            | (None, 333, 1024)      | 4198400 | ['lambda_9[0][0]']                                      |
| bidirectional_15 (Bidirectional)            | (None, 333, 1024)      | 6295552 | ['bidirectional_14[0][0]']                              |
| dense_7 (Dense)                             | (None, 333, 142)       | 145550  | ['bidirectional_15[0][0]']                              |

Total params: 12969038 (49.47 MB)
Trainable params: 12966478 (49.46 MB)
Non-trainable params: 2560 (10.00 KB)

Train this model on 100 epoch:
Epoch 68/100
68/68 [==============================] - ETA: 0s - loss: 0.2328
Epoch 68: val_loss did not improve from 10.65264
Restoring model weights from the end of the best epoch: 48.

Epoch 68: ReduceLROnPlateau reducing learning rate to 8.000000525498762e-06.
68/68 [==============================] - 114s 2s/step - loss: 0.2328 - val_loss: 10.8030 - lr: 4.0000e-05
Epoch 68: early stopping

Best model found with loss: 0.2328, val_loss 10.8030

Metrics:
Character Error Rate: 0.05310506221129512
Word Error Rate:      0.16521458479791834
Sequence Error Rate:  0.6555555555555556


## Key findings

- The model was lightweight 49.47 MB < 50 MB
- The model has inference time 10.7s, not fast enough
- The model has high error rate than expected

## Possible Flaws and Future Plans

### Possible Flaws

Model Complexity: While the model is lightweight, it might not be complex enough to capture all the nuances of handwritten Vietnamese text. The relatively high error rates suggest that the model might be underfitting the data.

Inference Time: The model’s inference time is currently 10.7 seconds, which exceeds the desired threshold of 2 seconds. This could be due to the complexity of the model or the size of the input data.

Data Quality and Quantity: The quality and quantity of the training data can significantly impact the performance of a deep learning model. If the training data is not representative of the data the model will encounter in real-world scenarios, or if there’s not enough data, the model might not perform well.

### Future Plans

Experiment with Different Model Architectures: I will try different model architectures or variations of existing ones. For example, different types of RNNs like GRU or use attention mechanisms.

Optimize Model for Inference Time: I will look into techniques for model optimization that can help reduce inference time, such as quantization, pruning, or using a more efficient model architecture.

Augment Training Data: Data augmentation techniques such as random rotations, shifts, and flips can create variations in the training data and help improve the model’s robustness and performance.

Use Pretrained Models: Transfer learning from pretrained models on similar tasks could potentially improve performance and reduce training time.

Hyperparameter Tuning: Experimenting with different hyperparameters like learning rate, batch size, number of layers, number of units per layer etc., can also lead to improvements in model performance.