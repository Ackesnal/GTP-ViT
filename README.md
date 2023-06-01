# Token Summarisation for Efficient Vision Transformers via Graph-based Token Propagation

# Usage

## Environment installation

Install Python 3.8+, PyTorch 1.7.0+, torchvision 0.8.1+ and timm==0.4.12:

```
conda create -n gtp python=3.8
conda activate gtp
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install timm==0.9.2
pip install torchprofile
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

Download DeiT pre-trained models.

## Evaluation

To evaluate GTP without training, run:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 --use_env main.py --data-path /path/to/imagenet/ --batch-size 256 --model graph_propagation_deit_small_patch16_224 --eval --resume /path/to/deit_small_patch16_224-cd65a155.pth --sparsity 1.0 --alpha 0.1 --num_prop 4 --selection MixedAttnMax --propagation GraphProp
```

You can adjust the arguments to explore difference results. Available values for these arguments can be found in main.py.