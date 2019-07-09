# VQA Papers #

A list of Visual Question Answering, Visual Dialog and Visual Reasoning papers.

## Survey ##
* [Survey of Visual Question Answering: Datasets and Techniques](https://arxiv.org/pdf/1705.03865) - *Gupta, Akshay Kumar. **arXiv** preprint arXiv:1705.03865. 2017.*
  * overview of some VQA datasets and comparison of some techniques on the DAQUAR and VQA datasets.
  * description of attention-based models, neural module networks and knowledge base VQA
* [Visual question answering: Datasets, algorithms, and future challenges](https://arxiv.org/abs/1610.01465) - *Kafle, Kushal, and Christopher Kanan. Computer Vision and Image Understanding (**CVIU**). 2017.*
   * overview of DAQUAR, COCO-QA, VQA, FM-IQA, Visual Genome, Visual7W, SHAPES
   * discussion of VQA metrics especially for open ended vs multiple choice
   * comparison of various networks on DAQUAR and COCO-QA
* [Visual question answering: A survey of methods and datasets](https://arxiv.org/abs/1607.05910) -  *Wu, Qi, Damien Teney, Peng Wang, Chunhua Shen, Anthony Dick, and Anton van den Hengel. Computer Vision and Image Understanding (**CVIU**). 2017.*
  * detailed description of neural module networks and dynamic memory networks
  * additionally to VQA datasets on natural images (DAQUAR, COCO-QA, VQA, FM-IQA, Visual Genome, Visual7W), an overview of datasets using clipart images is presented (abstract scenes, shapes) and knowledge base-enhanced datasets (KB-VQA and FVQA) 

## Datasets ##
### 1. Geometric Forms ###
* **CLEVR** - [Clevr: A diagnostic dataset for compositional language and elementary visual reasoning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.pdf) - *Johnson, Justin, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Conference on Computer Vision and Pattern Recognition (**CVPR**). 2017. *
   * 100K images with 1M questions.
   * in 3D space: 3 types of shapes (sphere, cube and cylinder), 2 sizes (small, large), 8 colors and 2 materials 
   * visual graph included for training and validation set (i.e., location and type of the forms and relationship between them: left, right, behind and in front). Download the predicted visual graph on the test set [here](https://cvhci.anthropomatik.kit.edu/~mhaurile/data/). 
* **SHAPES** - [Neural Module Networks](http://openaccess.thecvf.com/content_cvpr_2016/papers/Andreas_Neural_Module_Networks_CVPR_2016_paper.pdf) - *Andreas, Jacob, Marcus Rohrbach, Trevor Darrell, and Dan Klein. In Conference on Computer Vision and Pattern Recognition (**CVPR**). 2016. *
   * 15K questions on 64 images
   * 2D world in 3x3 grid with various instance types: 3 different colors (red, green, blue) and 3 shapes (circle, square and triangle)
* **Sort-of-CLEVR** - [A simple neural network module for relational reasoning](https://papers.nips.cc/paper/7082-a-simple-neural-network-module-for-relational-reasoning.pdf) - *Santoro, Adam, David Raposo, David G. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. In Advances in neural information processing systems (**NIPS**). 2017.*
   * 10K images with 200K questions
   * 6 objects per image with 2 shape types (circle, square) and 6 different colors
   * 10 relational and 10 non-relational question templates
* **COG** - [A dataset and architecture for visual reasoning with a working memory](https://arxiv.org/pdf/1803.06092) - *Yang, Guangyu Robert, Igor Ganichev, Xiao-Jing Wang, Jonathon Shlens, and David Sussillo. In European Conference on Computer Vision (**ECCV**). 2018.*
  * 4 to 8 frames per video
  * objects in 2D space with 19 possible colors and 33 possible shapes 
  * contains pointing and conditional (if and else) questions
  
  
### 2. Natural Images ###
* **VQA-v2**
* **GQA**
* **DAQUAR**
* **COCO-QA** - [Exploring models and data for image question answering](http://papers.nips.cc/paper/5640-exploring-models-and-data-for-image-question-answering.pdf) - Ren, Mengye, Ryan Kiros, and Richard Zemel. In Advances in neural information processing systems (**NIPS**). 2015.
  * generates questions from [MS-COCO](http://cocodataset.org/) captions using the Stanford parser 
  * 69K images and 118K questions
  * 4 types of questions: object (what), number, color and location (using the preposition "in")
* **VQA**


### 3. Videos ###
* **MovieQA** - [Movieqa: Understanding stories in movies through question-answering](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Tapaswi_MovieQA_Understanding_Stories_CVPR_2016_paper.pdf) - *Tapaswi, Makarand, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, and Sanja Fidler.In Conference on Computer Vision and Pattern Recognition (**CVPR**). 2016.*
  * 15K questions about 408 movies
  * multiple choice answers 
  * additionally to the videos, it includes plots, subtitles and scripts  
* **COG** - [A dataset and architecture for visual reasoning with a working memory](https://arxiv.org/pdf/1803.06092) - *Yang, Guangyu Robert, Igor Ganichev, Xiao-Jing Wang, Jonathon Shlens, and David Sussillo. In European Conference on Computer Vision (**ECCV**). 2018.*
  * 4 to 8 frames per video
  * objects in 2D space with 19 possible colors and 33 possible shapes 
  * contains pointing and conditional (if and else) questions

### 4. Embodied VQA Datasets ###
TODO

### 5. Other ###
TODO

## Methods ##

### 1. Global Emedding Techniques ###
* **Ask-your-neurons** - [Ask Your Neurons: A Neural-Based Approach to Answering Questions About Images](http://openaccess.thecvf.com/content_iccv_2015/papers/Malinowski_Ask_Your_Neurons_ICCV_2015_paper.pdf) - Mateusz Malinowski, Marcus Rohrbach, Mario Frit. In Conference on Computer Vision and Pattern Recognition (**CVPR**). 2015.
  * image represented as a vector produced by a pre-trained CNN
  * encodes the question using an LSTM
  * after the "end" token is added into the LSTM, the LSTM starts generating the answer
  * evaluated on DAQUAR
* [Learning to answer questions from image using convolutional neural network](https://arxiv.org/pdf/1506.00333.pdf) - Ma, Lin, Zhengdong Lu, and Hang Li. In Association for the Advancement of Artificial Intelligence (**AAAI**). 2016.
  * image represented using a pre-trained CNN  (from fully-connected layer --> vector representation = D)
  * question embedded using an 1D convolutional and maxpooling layers (representation=#words x D)
  * a multi-modal convolution layer imployed to project the two representations into same space
  * the multi-modal convolution layer has a kernel size of 3x1 
  * each convolution operation gets as input two neighboring vectors from the question matrix and the image vector (input = 3xD)
  * thus, the output of this layer is again a matrix (output = #words x H)
  * evaluated on DAQUAR and COCO-QA

### 2. Attention-Based Models ###
* **SAN** - [Stacked attention networks for image question answering](http://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Stacked_Attention_Networks_CVPR_2016_paper.pdf) - Yang, Zichao, Xiaodong He, Jianfeng Gao, Li Deng, and Alex Smola. In Conference on Computer Vision and Pattern Recognition (**CVPR**). 2016. 
  * question embedded using either an LSTM or a (1D) CNN. 
  * image represented using features of a pre-trained CNN
  * multi-step reasoning by attending to the image more than once (2 steps were proven best)
  * evaluated on DAQUAR, COCO-QA and VQA

  
  
### 3. Compositional Models ###
TODO

### 4. Memory Networks ###
TODO

### 5. Graph Nets ###
TODO

### 6. Incorporating Knowledge-bases ###

### 7. Embodied VQA ###

