## Paper Abstract

Diffusion models have demonstrated exceptional performance in image generation modeling. This paper proposes a novel semantic segmentation method based on diffusion models. By modifying the training and sampling schemes, we demonstrate that diffusion models can be applied to lesion segmentation in medical images. To generate segmentation results for specific images, we use real segmentation labels as targets during training and incorporate the original images as prior information in both the training and sampling processes. Due to the stochastic nature of the sampling process, we can generate a distribution of segmentation masks. This characteristic allows us to compute pixel-wise uncertainty maps for segmentation and enhance segmentation performance through implicit ensemble methods by integrating multiple segmentation results. We evaluate this method on the BRATS2020 dataset, and the results show not only excellent segmentation performance but also detailed uncertainty maps.


## Data

We evaluated our method on the [BRATS2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html).
For our dataloader, which can be found in the file *guided_diffusion/bratsloader.py*, the 2D slices need to be stored in the following structure:

```
data
└───training
│   └───BraTS20_Training_333
│       │   BraTS20_Training_333_flair.nii.gz
│       │   BraTS20_Training_333_seg.nii.gz
│       │   BraTS20_Training_333_t1.nii.gz
│       │   BraTS20_Training_333_t1ce.nii.gz
│       │   BraTS20_Training_333_t2.nii.gz
│   └───BraTS20_Training_334
│       │  ...
└───testing
│   └───BraTS20_Training_335
│       │   BraTS20_Training_335_flair.nii.gz
│       │   BraTS20_Training_335_t1.nii.gz
│       │   BraTS20_Training_335_t1ce.nii.gz
│       │   BraTS20_Training_335_t2.nii.gz
│   └───BraTS20_Training_336
│       │  ...

```

A mini-example can be found in the folder *data*.
If you want to apply our code to another dataset, make sure the loaded image has attached the ground truth segmentation as the last channel.


## Usage

We set the flags as follows:
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
```
To train the segmentation model, run

```
python3 scripts/segmentation_train.py --data_dir ./data/training $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```
The model will be saved in the *model* folder.
For sampling an ensemble of 5 segmentation masks with the DDPM approach, run:

```
python scripts/segmentation_sample.py  --data_dir ./data/testing  --model_path ./results/savedmodel.pt --num_ensemble=5 $MODEL_FLAGS $DIFFUSION_FLAGS
```
The generated segmentation masks will be stored in the *results* folder. A visualization of the sampling process is done using [Visdom](https://github.com/fossasia/visdom).

```
