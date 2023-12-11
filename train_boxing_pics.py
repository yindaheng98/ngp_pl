import subprocess
path = "/volume/data/boxing/"
indexes = range(14, 72)
# spell = "python train.py     --root_dir {}frame{}     --dataset_name 'colmap'    --exp_name boxing{} --no_save_test  --num_epochs 20 --batch_size 16384 --lr 2e-2"

for index in indexes:
    try:
        subprocess.run(['python', 'train.py', '--root_dir', f'{path}frame{index}',
                        '--dataset_name', 'colmap', '--exp_name', f'boxing{index}',
                        '--no_save_test', '--num_epochs', '20', '--batch_size', '16384',
                        '--lr', '2e-2'])
        with open('./train_log.txt', 'a') as log:
            log.write(f'index {index} ran successfully\n')
    except Exception as e:
        with open('./train_log.txt', 'a') as log:
            log.write(f'index {index} failed {e}\n')
