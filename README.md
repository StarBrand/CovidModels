# Covid Model

To import code, generate a virtual environment. By using conda:

```bash
conda create -n ml-covid python=3.9
```

Cloning repository:

```bash
git clone https://github.com/u-genoma/CovidModels.git
```

Go to repository folder and install requirements on virtual environment

```bash
cd CovidModels
conda activate ml-covid
pip install -r requirenments.txt
```

And install the repository as develop, in case changes are needed:

```bash
cd code
python setup.py develop
cd ..
```
