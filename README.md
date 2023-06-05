# Experiments of "Assuming Locally Equal Calibration Errors for Non-Parametric Multiclass Calibration"

## Environment
To create the environment you need to have the conda package manager installed and run in the root directory of this project
```
conda env create -n LECE --file ENV.yml
conda activate LECE
```

## Synthetic data experiments

See the notebook in "notebooks/Synthetic data experiment and Figures 1-2.ipynb".

## Real data experiments

## Data
Download the logits in pickle format from https://github.com/markus93/NN_calibration.
Place the downloaded logit files into the folder "logits".
Needed files:
* probs_densenet40_c10_logits.p
* probs_densenet40_c100_logits.p
* probs_resnet110_c10_logits.p
* probs_resnet110_c100_logits.p
* probs_resnet_wide32_c10_logits.p
* probs_resnet_wide32_c100_logits.p

### Calibration
Every calibration method has its own script to run the experiment.

| Method       | Running the experiment |
|----------|---|
| Uncalibrated predictions | `python -m src.experiments.uncalibrated`|
| Temperature scaling | `python -m src.experiments.temp_scaling`|
|LECE and LECD calibration|`python -m src.experiments.lece`|
|Isotonic regression calibration|`python -m src.experiments.isotonic`|
|Matrix scaling|`python -m src.experiments.matrix_scaling`|
|Decision calibration|`python -m src.experiments.decision`|
|Gaussian process calibration|[See detailed instructions below](#gp-calibration)|
|Intra-order preserving functions |[See detailed instructions below](#iop-functions)|

Note that it is essential to first run the experiment for the uncalibrated predictions as other methods expect to read the saved output files of `src.experiments.uncalibrated` as their input.
For methods applied in composition with temperature scaling (LECE, LECD, isotonic and decision calibration), it is also essential to beforehand run `src.experiments.temp_scaling`.

#### GP calibration
To also include Gaussian calibration (GP) in the experiments, 
1. Create a new Python environment as guided in the original repository https://github.com/JonathanWenger/pycalib.
2. Activate the created Python environment and in the root of this project run
```
python -m src.experiments.gp_cal
```

#### IOP functions
To also include the diagonal subfamily of intra-order preserving (IOP) functions in the experiments:
1. Clone the original repository https://github.com/AmirooR/IntraOrderPreservingCalibration.
2. Follow the environment and experiment set up described in their repository.
3. For each of the 6 model-dataset combinations, run their experiments as described in their repository
```
python -u calibrate.py --exp_dir exp_dir/{dataset}/{model}/DIAG --singlefold True
python -u evaluate.py --exp_dir exp_dir/{dataset}/{model}/DIAG --save_logits True
```
4. Copy the saved files "scores.npy" and "logits.npy" for each model-dataset combination from "exp_dir\{dataset}\{model}\DIAG\fold_1" into the corresponding folders of this project "results/precomputed/iop_diag/{model}_{dataset}".
5. To make the experiment results compatible with the table generation code of this article, as a final step run in the root of this project
```
python -m src.experiments.iop
```
Note that our experiments use exactly the same model-dataset logits as the IOP experiments, which is why it is possible to use the original experiments code of the IOP article.

### Tables and Figures

The runtime analysis of LECE and LECD is provided in notebook "notebooks/Runtimes (Appendix A3).ipynb".

The tables of the real data experiment are created in notebook "notebooks/Tables 3-11.ipynb".

