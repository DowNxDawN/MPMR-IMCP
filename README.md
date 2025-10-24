
# MPMR-IMCP Project

## Overview

This repository contains two main modules: **MPMR** and **IMCP**. The project is designed for medical image analysis, feature extraction, model training, evaluation, and visualization, with a focus on patch-based and multi-modal pathology research.

## Folder Structure

```
MPMR/
    geojson_QuPath_to_json_Labelme.py
    Labelme_TypeCount_ImgSep.py
    MPMR_predict.py
    MPMR_train.py
    assets/
        ckpts/
            uni2/
                UNI2-h.pth
    uni/
        __init__.py
        downstream/
            __init__.py
            extract_patch_features.py
            utils.py
            eval_patch_features/
                __init__.py
                fewshot.py
                linear_probe.py
                logistic_regression.py
                metrics.py
                protonet.py
        get_encoder/
            __init__.py
            get_encoder.py
            models/
                __init__.py
                resnet50_trunc.py
                model_wrappers/
                    __init__.py
                    timm_avgpool.py
    weights/
        CD10.pth
        MUC2.pth
        MUC5AC.pth
        MUC6.pth
IMCP/
    case_control.py
    feature_selection.py
    main.py
    model_evaluation.py
    model_explanation.py
    model_training.py
    model_tuning.py
    visualization.py
    logs/
    results/
```

## Modules

### MPMR

- **geojson_QuPath_to_json_Labelme.py**: Convert QuPath GeoJSON annotations to Labelme format.
- **Labelme_TypeCount_ImgSep.py**: Count label types and separate images for Labelme.
- **MPMR_predict.py**: Prediction pipeline for MPMR models.
- **MPMR_train.py**: Training pipeline for MPMR models.
- **assets/ckpts/uni2/**: Pretrained model checkpoints.
- **uni/**: Core library for feature extraction and model utilities.
  - **downstream/**: Downstream tasks and evaluation.
  - **get_encoder/**: Encoder models and wrappers.
- **weights/**: Pretrained weights for various markers.

### IMCP

- **case_control.py**: Case-control study utilities.
- **feature_selection.py**: Feature selection methods.
- **main.py**: Main entry point for IMCP workflows.
- **model_evaluation.py**: Model evaluation scripts.
- **model_explanation.py**: Model interpretability and explanation tools.
- **model_training.py**: Model training scripts.
- **model_tuning.py**: Hyperparameter tuning utilities.
- **visualization.py**: Visualization tools for results and analysis.
- **logs/**: Log files.
- **results/**: Output results.

## Requirements

- Python 3.7+
- Common libraries: numpy, pandas, scikit-learn, torch, torchvision, matplotlib, etc.
- (Optional) QuPath, Labelme for annotation processing.

## Usage

### Step 1: Install Dependencies
Install only the dependencies required for your specific tasks. Refer to the code for actual requirements.

### Step 2: Image Segmentation and Annotation
- For image segmentation and annotation using Labelme, run `Labelme_TypeCount_ImgSep.py` to generate annotation files based on Labelme format.
- If you use QuPath for annotation, you can convert QuPath GeoJSON files to Labelme format using `geojson_QuPath_to_json_Labelme.py`.

### Step 3: MPMR Model Training and Validation
- Run `MPMR_train.py` to train and validate the MPMR model. All configurable parameters are defined inside the script.
- Required pretrained encoder weights (UNI v2) should be placed at `MPMR/assets/ckpts/uni2/UNI2-h.pth` before training.
    - Note: Due to GitHub file size limits, `MPMR/assets/ckpts/uni2/UNI2-h.pth` is not included in this repository. Obtain the UNI v2 pretrained weights from the official UNI repository: https://github.com/mahmoodlab/UNI
- Trained model weights referenced by this project are expected under `MPMR/weights/`.
    - Note: Trained weights are not included here because of GitHub size restrictions. Download the trained model weights from our Google Drive: https://drive.google.com/file/d/1ivoxS5H19G7wxcgK4vE7j23SoQQ1SmZU/view?usp=sharing

### Step 4: MPMR Model Prediction and Heatmap Generation
- Use `MPMR_predict.py` for model prediction and heatmap generation. All parameter settings are included in the script.

### Step 5: IMCP Model Usage
- Run `main.py` in the `IMCP/` directory to use the IMCP model. Specific parameters can be modified in the corresponding sub-scripts as needed.

## Citation

If you use this codebase in your research, please cite the corresponding paper or acknowledge the authors.

## License

This project is for academic research purposes only. For commercial use, please contact the authors.
