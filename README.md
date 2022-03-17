# MIL

## Training

We suggest putting all data following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– tile-dataset/
|   |–– train/ 
|   |   |-- tumor/
|   |   |-- normal/
|   |–– val/
|   |   |-- tumor/
|   |   |-- normal/
```

To train model, run:
```
python train_qresnet.py --data-loader from_folder --training-img-dir path/to/train/data --val-img-dir path/to/val/data
```