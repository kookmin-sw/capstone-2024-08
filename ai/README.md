## environment
```bash
conda create -n loro python=3.10
conda activate loro

cd vits2_pytorch
sudo apt-get install espeak
sudo pip3 install --ignore-installed -r requirements.txt
```

## vscode에서 Jupyter notebook 사용하는 경우
```bash
pip3 install jupyter
pip3 install ipykernel
python3 -m ipykernel install --user --name loro --display-name loro
```


## dataset
```bash
ln -s /path/to/dataset/VL DUMMY3

```