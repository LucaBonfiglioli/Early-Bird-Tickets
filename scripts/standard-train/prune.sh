python vggprune.py \
--dataset cifar100 \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/vgg16-cifar100/ckpt1_30.pth.tar \
--save ./baseline/vgg16-cifar100/test_3035_0.3 \
--gpu_ids 0