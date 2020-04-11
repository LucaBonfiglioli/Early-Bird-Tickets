import os

lr = '0.1'
save = 'vgg16-cifar100_lf'
pr_list = ['30', '50', '70']
snap_list = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150']

base_search = 'CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr '+lr+' \
--epochs 160 \
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
--epochs 160 \
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
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/'+save+'/EB_retrain%d_%d_'+lr+' \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/'+save+'/EB_prune%d_%d/pruned.pth.tar \
--start-epoch %d'

print('EXECUTING SEARCH')
os.system(base_search)
for pr in pr_list:
    pr = int(pr)
    for snap in snap_list:
        snap = int(snap)
        print('RETRAINING PR %d AND SNAP %d' % (pr, snap))
        os.system(base_prune % (snap, pr, snap, pr))
        os.system(base_retrain % (snap, pr, snap, pr, snap))
    files = os.listdir('/baseline/'+save)
    b = []
    for file in files:
        if 'EB' in file and '_m.' not in file and file[3:5] == pr:
            b.append(file)
    if len(b) == 1:
        b = b[0]
    snap = int(b.split('_')[2][0:-8])
    print('RETRAINING EB PR %d' % pr)
    os.system(base_eb_prune % (snap, pr, snap, pr))
    os.system(base_eb_retrain % (snap, pr, snap, pr, snap))