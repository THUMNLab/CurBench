from .base import BaseTrainer
from .spl import SPL



class TTCL(SPL):
    """Transfer Teacher Curriculum Learning.

    It is inherited from the Self-Paced Learning, but the data difficulty is decided by a pre-trained teacher. 
    Curriculum learning by transfer learning: Theory and experiments with deep networks. http://proceedings.mlr.press/v80/weinshall18a/weinshall18a.pdf
    A Survey on Curriculum Learning. https://arxiv.org/pdf/2010.13166.pdf

    Attributes:
        name, dataset, data_size, batch_size, n_batches: Base class attributes.
        epoch, start_rate, grow_epochs, grow_fn, device, criterion, weights: SelfPaced class attributes.
        net: A pre-trained teacher net.
        data_loss: save the loss calculated by the teacher net.
    """
    def __init__(self, start_rate, grow_epochs, grow_fn, weight_fn, teacher_net):
        super(TTCL, self).__init__(
            start_rate, grow_epochs, grow_fn, weight_fn)

        self.name = 'ttcl'
        self.teacher_net = teacher_net
        self.data_loss = None


    def _loss_measure(self):
        """Only calculate the data loss once, because the teacher net is fixed and the loss will not be changed."""
        if self.data_loss is None:
            self.data_loss = super()._loss_measure()
        return self.data_loss



class TTCLTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, 
                 start_rate, grow_epochs, grow_fn, weight_fn, teacher_net):

        cl = TTCL(start_rate, grow_epochs, grow_fn, weight_fn, teacher_net)

        super(TTCLTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)