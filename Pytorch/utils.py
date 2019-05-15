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
		val *= self.loss_aug
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
