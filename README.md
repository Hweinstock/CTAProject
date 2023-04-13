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
- [TinyBert](https://arxiv.org/abs/2110.01518)  
This paper outlines different versions of the Bert model and how the number of parameters affects the performance on different datasets. 
- [DistillBert](https://arxiv.org/abs/1910.01108)  
This paper provides a smaller version of Bert to work with that rivals performance of original. 
- [Stock Movement Prediction From Tweets and Historical Prices](https://homepages.inf.ed.ac.uk/scohen/acl18stock.pdf)  
This is the core paper that the model is generally based on. 
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)  
This paper gives examples of what learning rates work best. 
- [Sharp Ratio](https://www.seputarforex.com/belajar/forex_ebook/download/mahir/The_Sharpe_Ratio.pdf)
This paper outlines the modern method for evaluating trading strategies taking into account both risk and return, where risk is defined relative to maximum drawdown (MD). 
- [Review of Deep Learning in Stock Prediction](https://arxiv.org/abs/2003.01859)  
This paper outlines the different techniquess used and outlines some critical general strategies such as feature selection, PCA to reduce data redundancy, feature testing, and common evaluation techniques. 
- [NLP in finance forecasting](https://link.springer.com/article/10.1007/s10462-017-9588-9)
This paper gives an overview of techniques used in the field and the emmerging techniques as well. It suggests general flows for financial forecasting as well as outlines common data sources and their effects. 
