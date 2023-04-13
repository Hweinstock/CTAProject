# CTA Final Project

## Running
[colab notebook:](https://colab.research.google.com/drive/1AI5gzLZZMYNEnkfOxdt-SxgrHiEJ77pH#scrollTo=hsXTl74PrikK)
## Setup 

clone the repository
build the docker
`docker-compose build cta`
run the docker 
`docker-compose run --rm cta`
In the data_tools dir:
`python3 main.py`

Some papers we use:  
*[TinyBert](https://arxiv.org/abs/2110.01518)  
This paper outlines different versions of the Bert model and how the number of parameters affects the performance on different datasets. 
*[DistillBert](https://arxiv.org/abs/1910.01108)  
This paper provides a smaller version of Bert to work with that rivals performance of original. 
*[Stock Movement Prediction From Tweets and Historical Prices](https://homepages.inf.ed.ac.uk/scohen/acl18stock.pdf)  
This is the core paper that the model is generally based on. 
*[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)  
This paper gives examples of what learning rates work best. 

