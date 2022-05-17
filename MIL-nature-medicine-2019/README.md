# MIL-nature-medicine-2019
This repository provides training and testing scripts for the article *Campanella et al. 2019*.

## Weakly-supervised tile level classifier

### MIL Input Data
Input data, whether for training, validation or testing, should be a dictionary stored to disk with `torch.save()` containing the following keys:
* `"slides"`: list of full paths to WSIs (e.g. `my/full/path/slide01.svs`). Size of the list equals the number of slides.
* `"grid"`: list of a list of tuple (x,y) coordinates. Size of the list is equal to the number of slides. Size of each sublist is equal to the number of tiles in each slide. An example grid list containing two slides, one with 3 tiles and one with 4 tiles:
```python
grid = [
        [(x1_1, y1_1),
	 (x1_2, y1_2),
	 (x1_3, y1_3)],
	[(x2_1, y2_1),
	 (x2_2, y2_2),
	 (x2_3, y2_3),
	 (x2_4, y2_4)],
]
```
* `"targets"`: list of slide level class (0: benign slide, 1: tumor slide). Size of the list equals the number of slides.
* `"mult"`: scale factor (float) for achieving resolutions different than the ones saved in the WSI pyramid file. Usually `1.` for no scaling.
* `"level"`: WSI pyramid level (integer) from which to read the tiles. Usually `0` for the highest resolution.

To create input data, you organize your dataset :
```
$DATA/
|–– slide-dataset/
|   |–– tumor_slides/ 
|   |   |-- B05.4754_E.mrxs/
|   |   |-- B09.10583_VI E.mrxs/
|   |–– normal_slides/
|   |   |-- /
|   |   |-- /
```
And then, run `python MIL-nature-medicine-2019/prepare_data_file.py` to create input file to train MIL and RNN.

### MIL Training
To train a model, use script `MIL_train.py`. Run `python MIL_train.py -h` to get help regarding input parameters. Set `True` for `--use_quantum` if you want to use `DressedQuantum` layer.

```
python MIL_train.py --train_lib path/to/data/file/tile.pth --val_lib path/to/data/file/tile.pth --k 100 --test_every 1 --use_quantum True
```

Script outputs:
* **convergence.csv**: *.csv* file containing training loss and validation error metrics.
* **checkpoint_best.pth**: file containing the weights of the best model on the validation set. This file can be used with the `MIL_test.py` script to run the model on a test set. In addition, this file can be used to generate the embedding needed to train the RNN aggregator.

### MIL Testing
To run a model on a test set, use script `MIL_test.py`. Run `python MIL_test.py -h` to get help regarding input parameters.

```
python MIL_test.py --lib path/to/data/file/tile.pth --model path/to/checkpoint/MIL-nature-medicine-2019/checkpoint_best.pth --use_quantum True
```

Script outputs:
* **predictions.csv**: *.csv* file with slide name, slide target, model prediction and tumor probability entries for each slide in the test data. This file can be used to generate confusion matrix, ROC curve and AUC.

## RNN Aggregator

### RNN Input Data
Input data, whether for training, validation or testing, should be a dictionary stored to disk with `torch.save()` containing the following keys:
* `"slides"`: list of full paths to WSIs (e.g. `my/full/path/slide01.svs`). Size of the list equals the number of slides.
* `"grid"`: list of a list of tuple (x,y) coordinates. Size of the list is equal to the number of slides. Size of each sublist is equal to the number of maximum number of recurrent steps (we used 10). Each sublist is in decreasing order of tumor probability.
* `"targets"`: list of slide level class (0: benign slide, 1: tumor slide). Size of the list equals the number of slides.
* `"mult"`: scale factor (float) for achieving resolutions different than the ones saved in the WSI pyramid file. Usually `1.` for no scaling.
* `"level"`: WSI pyramid level (integer) from which to read the tiles. Usually `0` for the highest resolution.

### RNN Training
To train the RNN aggregator model, use script `RNN_train.py`. Run `python RNN_train.py -h` to get help regarding input parameters. You will need to have a trained embedder using the script `MIL_train.py`.

```
python RNN_train.py --train_lib path/to/data/file/tile.pth --val_lib path/to/data/file/tile.pth --model path/to/checkpoint/MIL-nature-medicine-2019/checkpoint_best.pth --use_quantum_cnn True --use_quantum_rnn True
```

Script outputs:
* **convergence.csv**: *.csv* file containing training loss and validation error metrics.
* **rnn_checkpoint_best.pth**: file containing the weights of the best model on the validation set. This file can be used with the `RNN_test.py` script.

### RNN Testing

```
python RNN_test.py --lib path/to/data/file/tile.pth --model path/to/checkpoint/MIL-nature-medicine-2019/checkpoint_best.pth --rnn path/to/checkpoint/MIL-nature-medicine-2019/rnn_checkpoint_best.pth
```

To run a model on a test set, use script `RNN_test.py`. Run `python RNN_test.py -h` to get help regarding input parameters.
Script outputs:
* **predictions.csv**: *.csv* file with slide name, slide target, model prediction and tumor probability entries for each slide in the test data. This file can be used to generate confusion matrix, ROC curve and AUC.
