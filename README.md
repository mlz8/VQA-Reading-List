# VQA Paper #

A list of Visual Question Answering, Visual Dialog and Visual Reasoning papers.

## Survey ##
* [Survey of Visual Question Answering: Datasets and Techniques](https://arxiv.org/pdf/1705.03865) - *Gupta, Akshay Kumar. arXiv preprint arXiv:1705.03865. 2017. *
  * overview of some VQA datasets and comparison of some techniques on the DAQUAR and VQA datasets.
  * description of attention-based models, neural module networks and knowledge base VQA
* [Visual question answering: Datasets, algorithms, and future challenges](https://arxiv.org/abs/1610.01465) - *Kafle, Kushal, and Christopher Kanan. Computer Vision and Image Understanding (CVIU). 2017.*
   * overview of DAQUAR, COCO-QA, VQA, FM-IQA, Visual Genome, Visual7W, SHAPES
   * discussion of VQA metrics especially for open ended vs multiple choice
   * comparison of various networks on DAQUAR and COCO-QA
* [Visual question answering: A survey of methods and datasets](https://arxiv.org/abs/1607.05910) -  *Wu, Qi, Damien Teney, Peng Wang, Chunhua Shen, Anthony Dick, and Anton van den Hengel. Computer Vision and Image Understanding (CVIU). 2017. *
  * detailed description of neural module networks and dynamic memory networks
  * additionally to VQA datasets on natural images (DAQUAR, COCO-QA, VQA, FM-IQA, Visual Genome, Visual7W), an overview of datasets using clipart images is presented (abstract scenes, shapes) and knowledge base-enhanced datasets (KB-VQA and FVQA) 

## Datasets ##
### 1. Geometric Forms ###
* **CLEVR** - [Clevr: A diagnostic dataset for compositional language and elementary visual reasoning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.pdf) - *Johnson, Justin, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Conference on Computer Vision and Pattern Recognition (CVPR). 2017.*
  * 100K images with 1M questions.
  * in 3D space: 3 types of shapes (sphere, cube and cylinder), 2 sizes (small, large), 8 colors and 2 materials 
  * visual graph included for training and validation set (i.e., location and type of the forms and relationship between them: left, right, behind and in front). Download the predicted visual graph on the test set [here](https://cvhci.anthropomatik.kit.edu/~mhaurile/data/). 


* **SHAPES** - [Neural Module Networks](http://openaccess.thecvf.com/content_cvpr_2016/papers/Andreas_Neural_Module_Networks_CVPR_2016_paper.pdf) - *Andreas, Jacob, Marcus Rohrbach, Trevor Darrell, and Dan Klein. In Conference on Computer Vision and Pattern Recognition (CVPR). 2016. *
  * 15K questions on 64 images
  * 2D world in 3x3 grid with various instance types: 3 different colors (red, green, blue) and 3 shapes (circle, square and triangle)
  
&nbsp;
  

* **Sort-of-CLEVR** - [A simple neural network module for relational reasoning](https://papers.nips.cc/paper/7082-a-simple-neural-network-module-for-relational-reasoning.pdf) - *Santoro, Adam, David Raposo, David G. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. In Advances in neural information processing systems (NIPS). 2017.*
  * 10K images with 200K questions
  * 6 objects per image with 2 shape types (circle, square) and 6 different colors
  * 10 relational and 10 non-relational question templates

&nbsp;

* **COG** - [A dataset and architecture for visual reasoning with a working memory](https://arxiv.org/pdf/1803.06092) - *Yang, Guangyu Robert, Igor Ganichev, Xiao-Jing Wang, Jonathon Shlens, and David Sussillo. In European Conference on Computer Vision (ECCV). 2018.*
  * 4 to 8 frames per video
  * objects in 2D space with 19 possible colors and 33 possible shapes 
  * contains pointing and conditional (if and else) questions
  
  
### 2. Natural Images ###
* **VQA**
* **VQA-v2**
* **GQA**


### 3. Videos ###

* **COG** - [A dataset and architecture for visual reasoning with a working memory](https://arxiv.org/pdf/1803.06092) - *Yang, Guangyu Robert, Igor Ganichev, Xiao-Jing Wang, Jonathon Shlens, and David Sussillo. In European Conference on Computer Vision (ECCV). 2018.*
  * 4 to 8 frames per video
  * objects in 2D space with 19 possible colors and 33 possible shapes 
  * contains pointing and conditional (if and else) questions

### 4. Embodied VQA Datasets ###
TODO

### 5. Other ###
TODO

## Methods ##

### 1. Global Emedding Techniques ###
TODO

### 2. Attention-Based Models ###
TODO

### 3. Compositional Models ###
TODO

### 4. Memory Networks ###
TODO

### 5. Graph Nets ###
TODO

### 6. Incorporating Knowledge-bases ###

### 7. Embodied VQA ###

