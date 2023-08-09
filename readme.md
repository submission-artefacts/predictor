# Description
This is a python package for serving the SVD model via REST APIs using Flask framework. Source code for SVD using Alternating Least Squares(ALS) can be found in `als_recommendation`. SVD using Stochastic Gradient Decent(SGD) is implemented using [scikit-surprise](https://surpriselib.com/)


# Deployment 
##
1. Go to the `predictor` directory and install all the necessary packages using:
```shell
cd predictor
pip install -r requirements.txt
```
2. Run the python app:

```shell
python app.py


python app.py