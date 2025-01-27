# automation_scripts
Here is a collection of a few automation scripts for speeding up the development of facefeatures

**buildlayer.py** is used for building out our lambda_function.py files based of a template

**tritonconfig.py** is used for automatically generating config.pbtxt files and packaging them along with your models into a model_repository.tar.gz file ready for the triton inference server to use

## How to use

These scripts are designed to be portable and simple to use.

```
clone https://github.com/tmasynthetics/automation_scripts
cd automation_scripts
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

## Navigate to your project directory and copy the script into the root of your project directory
## You must be able to access the models file from 'config/models.py' relative to your current working directory

python tritonconfig.py
```

The script will package the model_repository for you