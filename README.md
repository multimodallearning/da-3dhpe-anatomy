# da-3dhpe-anatomy
Source code for our MIDL2022 paper [Domain adaptive 3d human pose estimation through anatomical constraints](https://openreview.net/forum?id=iCTU7EQipC) [[pdf](https://openreview.net/pdf?id=iCTU7EQipC)]  and its ongoing journal extension [Anatomy-guided domain adaptation for 3D in-bed human pose estimation](https://arxiv.org/abs/2211.12193) [[pdf](https://arxiv.org/pdf/2211.12193.pdf)].

## Dependencies
Please first install the following dependencies
* Python3 (we use 3.8.3)
* numpy
* pytorch (we tested 1.6.0 and 1.9.0)
* yacs
* smplpytorch
* scipy
* cv2
* skimage

Furhtermore, please clone the following repositories to the same directory as our repository
* https://github.com/pgrady3/SLP-3Dfits
* https://github.com/ostadabbas/SLP-Dataset-and-Code

## Data Preparation
1. Download the SLP dataset from https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/. Create a directory `/datasets/SLP` and move move the dataset to this directory. We recommend to create a symlink.
2. Download the male and female [SMPL model](https://smpl.is.tue.mpg.de/). In `data/process_slp.py`, modify `smpl_path` in line 166 such that it points to the downloaded models.
3. Execute `cd data` and `python process_slp.py` to generate point clouds from the original depth images and to obtain the ground truth 3D joints. The data is written to `/datasets/SLP_processed`.

## Training
1. In `/configs/defaults.py`, modify `_C.BASE_DIRECTORY` in line 5 to the root directory where you intend to save the results.
2. In the config files `/configs/CONFIG_TO_SPECIFY.yaml`, you can optionally modify `EXPERIMENT_NAME` in line 1. Models and log files will finally be written to `os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)`.
3. Navigate to the `main` directory.
4. To train the source-only or target-only model, execute `python train.py --gpu GPU --config-file ../configs/config_baseline.yaml` or `python train.py --gpu GPU --config-file ../configs/config_oracle.yaml`. After each epoch, we save the model weights and a log file to the specified directory after each epoch.
5. To train our model as in the MIDL paper (UDA, anatomy-constrained optimization only), execute `python train.py --gpu GPU --config-file ../configs/config_pretrain.yaml` for pre-training and subsequently execute `python train.py --gpu GPU --config-file ../configs/config_ours.yaml`. Again, we save the model weights and a log file to the specified directory.
6. To train our model as in the journal extension (anatomy-constrained optimization & anatomy-guided PL filtering with the Mean Teacher), execute `python train_uda.py --gpu GPU --config-file ../configs/config_anat-mt_uda.yaml` for training in the UDA setting. Training in the SFDA setting requires previous training of the source-only model for initialization (see 4.). Alterantively, we provide the [weights](https://drive.google.com/file/d/1-p4jP7qJ42P0WdXS9BCv2sgwIbZoSbg4/view?usp=sharing) of our trained source-only model. Given the pretrained model, set `WEIGHT`in line 4 of `/configs/config_anat-mt_sfda.yaml` to the path of the weights and subsequently execute `python train_sfda.py --gpu GPU --config-file ../configs/config_anat-mt_sfda.yaml`.

## Testing
* If you trained a model yourself following the instructions above, you can test the model by executing `python test.py --config-file ../configs/CONFIG_USED_FOR_TRAINING.yaml --gpu GPU --val-split VAL_SPLIT --cover-condition COVER_COND --position POSITION`. `VAL_SPLIT`should be in {val, test}, `COVER_COND` should be in {cover1, cover2, cover12}, and `POSITION` should be in {supine, lateral, all, left, right}. The output is the MPJPE in mm, averaged over all joints as well as averaged over different groups of joints, for specified cover condition and patient position.
* Otherwise, we provide our [pre-trained models](https://drive.google.com/drive/folders/1cQ6owCgX3VG4lrt8CCzspLHEFW0321vi). Download the models and use it for inference by executing `python test.py --config-file ../configs/CONFIG_OF_CHOSEN_MODEL.yaml --gpu GPU --val-split VAL_SPLIT --cover-condition COVER_COND --position POSITION --model-path /PATH/TO/MODEL`. The models achieve the same results as reported in our paper.

## Citation
If you find our code useful for your work, please cite the following papers
```latex
@inproceedings{bigalke2021domain,
  title={Domain adaptation through anatomical constraints for 3d human pose estimation under the cover},
  author={Bigalke, Alexander and Hansen, Lasse and Diesel, Jasper and Heinrich, Mattias P},
  booktitle={Medical Imaging with Deep Learning},
  year={2021}
}

@article{bigalke2022anatomy,
  title={Anatomy-guided domain adaptation for 3D in-bed human pose estimation},
  author={Bigalke, Alexander and Hansen, Lasse and Diesel, Jasper and Hennigs, Carlotta and Rostalski, Philipp and Heinrich, Mattias P},
  journal={arXiv preprint arXiv:2211.12193},
  year={2022}
}
```

## Acknowledgements
* Code for data pre-processing has been adapted from https://github.com/pgrady3/SLP-3Dfits
* DGCNN implementation has been adapted from https://github.com/AnTao97/dgcnn.pytorch

We thank all authors for sharing their code!
