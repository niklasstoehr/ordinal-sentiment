# Sentiment as an Ordinal Latent Variable
_______________________________________________

Code accompanying the EACL 2023 paper
https://aclanthology.org/2023.eacl-main.8.pdf

## Set-up

```
cd /Users/username/Code/ordinal-sentiment 
virtualenv ordinal-sentiment venv
source ordinal-sentiment _venv/bin/activate
pip install ipykernel
python3 -m ipykernel install --user --name=ordinal-sentiment _venv 
pip install -r requirements.txt
```
In order to run the code locally, you have to install the user-defined modules ``s0configs``,``s1data``,``s2model`` and ``s3analysis``. From the root, install the modules by executing

```
user@MacBook-Pro ordinal-sentiment %
pip install -e s0configs
pip install -e s1data
pip install -e s2model
pip install -e s2analysis
```