pip install --upgrade pip
pip install -r demos/requirements.txt
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension
python download.py --quick_demo