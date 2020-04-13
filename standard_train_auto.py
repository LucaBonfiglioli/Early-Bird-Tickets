import os
import torch

lr = '0.1'
save = 'vgg16-cifar100_lf'
pr_list = [30, 50, 70]
snap_list = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159]
epochs_base = 160

torch.set_num_threads(6)

base_search = 'CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr '+lr+' \
--epochs %d \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 256 \
--save ./baseline/'+save+' \
--momentum 0.9 \
--sparsity-regularization'

base_prune = 'CUDA_VISIBLE_DEVIDES=0 python vggprune.py \
--dataset cifar100 \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/'+save+'/ckpt%d_%d.pth.tar \
--save ./baseline/'+save+'/prune%d_%d \
--gpu_ids 0'

base_retrain = 'CUDA_VISIBLE_DEVICES=0 python main_c.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr '+lr+' \
--epochs %d \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/'+save+'/retrain%d_%d_'+lr+' \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/'+save+'/prune%d_%d/pruned.pth.tar \
--start-epoch %d'

base_eb_prune = 'CUDA_VISIBLE_DEVIDES=0 python vggprune.py \
--dataset cifar100 \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/'+save+'/EB_%d_%d.pth.tar \
--save ./baseline/'+save+'/EB_prune%d_%d \
--gpu_ids 0'

base_eb_retrain = 'CUDA_VISIBLE_DEVICES=0 python main_c.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr '+lr+' \
--epochs %d \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/'+save+'/EB_retrain%d_%d_'+lr+' \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/'+save+'/EB_prune%d_%d/pruned.pth.tar \
--start-epoch %d'

print('SEARCHING')
os.system(base_search % epochs_base)
for pr in pr_list:
    for snap in snap_list:
        print('PRUNING PR %d AND SNAP %d' % (pr, snap))
        os.system(base_prune % (snap, pr, snap, pr))
        print('RETRAINING PR %d AND SNAP %d' % (pr, snap))
        os.system(base_retrain % (snap + epochs_base, snap, pr, snap, pr, snap))
    files = os.listdir('./baseline/'+save)
    b = []
    for file in files:
        if 'EB' in file and '_m.' not in file and file[3:5] == str(pr):
            b.append(file)
    if len(b) == 1:
        b = b[0]
    snap = int(b.split('_')[2][0:-8])
    print('PRUNING EB PR %d' % pr)
    os.system(base_eb_prune % (pr, snap, snap, pr))
    print('RETRAINING EB PR %d' % pr)
    os.system(base_eb_retrain % (snap + epochs_base, snap, pr, snap, pr, snap))