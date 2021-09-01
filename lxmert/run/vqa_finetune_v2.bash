# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa_combine_v2.py \
    --train train --valid valid  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --optim bert --lr 6e-5 --epochs 30 \
    --tqdm --output $output ${@:3} \
    --load_cls /content/drive/MyDrive/lxmert/lstmcnn_classifier/best_accuracy_log.pth \
    --alpha 1 --beta 1