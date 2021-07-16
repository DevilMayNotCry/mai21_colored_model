# Deep Learning for Smartphone ISP 

### Image directory structure

    dataset_dir | train  | raw
                |        | retouched
                | test   | raw
                         | retouched

### Jupyter

    cd /content/mai21-learned-smartphone-isp
  
    !pip install scipy==1.2.0
  
    !CUDA_VISIBLE_DEVICES=0 python train_model.py dataset_dir=<image_directory> num_maps_base=32 batch_size=1 patch_w=2048 patch_h=2048 arch=punet model_dir=<dir where model saved> result_dir=<result images> train_size=1000 vgg_dir=<path to vgg weights>
  
