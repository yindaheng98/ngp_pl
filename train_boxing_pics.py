import os
path = "/volume/data/boxing/"
indexes = range(5, 24)
spell = "python train.py     --root_dir {}frame{}     --dataset_name 'colmap'    --exp_name boxing{} --no_save_test  --num_epochs 20 --batch_size 16384 --lr 2e-2"
for index in indexes:
    index_spell = spell.format(path, index, index)
    os.system(index_spell)