# poi-prediction

## Preparing the environment

Create and activate your virtual environment, e.g. by running

```
conda create -n poi-prediction python=3.10
cona activate poi-prediction
```

Set the python interpreter path in your IDE to the version in the environment, i.e. the output of

```
which python
```

Install the BIDS toolbox as provided in this repository

```
unzip bids.zip
cd bids
python setup.py build
python setup.py install
```

Back in the project directory, install the required packages 

```
pip install -r requirements.txt
```


## Preparing Data



