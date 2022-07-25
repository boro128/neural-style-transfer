#!/bin/bash

# Example usage:
# ./grid_search.sh images/style/The_Starry_Night_van_Gogh_240p.jpg images/Tuebingen_Neckarfront.jpg

if [ -z $1 ]
then
    echo Style image not specified
    exit 1
else
    style_img_path=$1
fi

if [ -z $2 ]
then
    echo Content image not specified
    exit 1
else
    content_img_path=$2
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate style-transfer

# feel free to adjust parameters below
height=240
width=320
save_freq=100

# change values below to adjust search space
lrs=(0.01 0.1)
alphas=(0.1)
betas=(8000000 7000000 6000000 5000000 4000000 30000000)

save_dir_prefix=nst/grid_search_`date +%Y-%m-%d_%H-%M`

for lr in ${lrs[@]}
do
    for alpha in ${alphas[@]}
    do
        for beta in ${betas[@]}
        do
            ~/anaconda3/envs/style-transfer/bin/python neural_style_transfer.py \
            --style-img-path  $style_img_path \
            --content-img-path $content_img_path \
            --height $height --width $width \
            --save-freq $save_freq \
            --lr $lr --alpha $alpha --beta $beta \
            --save-dir ${save_dir_prefix}/lr_${lr}_alpha_${alpha}_beta_${beta} \
            --resize-style-img
        done
    done
done
