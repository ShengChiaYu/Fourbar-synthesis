import argparse

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, loss_aug=1):
		self.reset()
		self.loss_aug = loss_aug

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		# val *= self.loss_aug
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fourbar network')
    # Load data
    parser.add_argument('--data_dir', dest='data_dir',
                        help='directory to dataset',
                        default="data", type=str)
    parser.add_argument('--pos_num', dest='pos_num',
                        help='60, 360',
                        default='60positions', type=str)
    parser.add_argument('--target', dest='tar',
                        help='param, pos',
                        default='_param.csv', type=str)
    parser.add_argument('--img_size', dest='img_size',
                        help='image size',
                        default=224, type=int)
    parser.add_argument('--class_num', dest='class_num',
                        help='class number',
                        default=4, type=int)
    parser.add_argument('--select_dir', dest='select_dir',
                        help='select_dir',
                        default='GCRR', type=None)

    # Training setup
    parser.add_argument('--net', dest='net',
                        help='Net_1, LeNet, resnet50, resnet152',
                        default='LeNet', type=str)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='True,False',
                        default=False, type=bool)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=100, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default='models', type=str)

    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=6, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=32, type=int)

    parser.add_argument('--cuda', dest='use_cuda',
                        help='whether use CUDA',
                        default=False, type=bool)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU id to use.',
                        default=0, type=int)

    parser.add_argument('--threshold', dest='threshold',
                        help='threshold of correctness',
                        default=0.01, type=float)
    parser.add_argument('--patience', dest='patience',
                        help='maximum count of early stopping',
                        default=5, type=int)


    # Configure optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # Predict setup
    parser.add_argument('--model_name', dest='model_name',
                        help='predicted model name',
                        default='model', type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory of predicted model',
                        default='models', type=str)

    args = parser.parse_args()

    return args
