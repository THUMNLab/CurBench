from collections import deque
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data.batch import Batch as pygBatch

from .base import BaseTrainer, BaseCL


def estimate_slope(x, y):
    assert len(x) == len(y)
    A = np.vstack([x, np.ones(len(x))]).T
    c, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return c



class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon


    def __call__(self, Q):
        # find the best action with random tie-breaking
        idx = np.where(Q == np.max(Q))[0]
        assert len(idx) > 0, str(Q)
        a = np.random.choice(idx)

        # create a probability distribution
        p = np.zeros(len(Q))
        p[a] = 1

        # Mix in a uniform distribution, to do exploration and
        # ensure we can compute slopes for all tasks
        p = p * (1.0 - self.epsilon) + self.epsilon / p.shape[0]

        assert np.isclose(np.sum(p), 1)
        return p



class ThompsonPolicy(EpsilonGreedyPolicy):
    pass


class BoltsmannPolicy:
    def __init__(self, temperature=1.) -> None:
        self.temperature = temperature

    def __call__(self, Q):
        e = np.exp((Q-np.max(Q))/self.temperature)
        p = e/np.sum(e)
        assert np.isclose(np.sum(p), 1)
        return p


class RLTeacherOnline(BaseCL):
    """Reinforcement Learning Teacher CL Algorithm. 

    Teacher-student curriculum learning. https://arxiv.org/pdf/1707.00183
    """
    def __init__(self, ):
        super(RLTeacherOnline, self).__init__()

        self.name = 'rl_teacher_online'
        self.policy = EpsilonGreedyPolicy(0.01)

        self.catnum = 10
        self.alpha = 0.1
        self.total = [0 for _ in range(self.catnum)]
        self.abs = False

        self.accs = [0 for _ in range(self.catnum)]
        self.reward = []


    def data_split(self):
        self.indexs = [range(i * self.data_size // self.catnum, (i + 1) * self.data_size // self.catnum) 
                       for i in range(self.catnum)]
        l = [len(self.indexs[i]) // self.catnum for i in range(self.catnum)]
        self.data = [self._dataloader(Subset(self.dataset, self.indexs[i]))
                     for i in range(self.catnum)]
        self.validationData = [self._dataloader(Subset(self.dataset, self.indexs[i][:l[i]]))
                               for i in range(self.catnum)]


    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)
        self.metric = kwargs.get("metric")
        self.metric_name = kwargs.get("metric_name")
        self.data_split()


    def data_curriculum(self, **kwargs):
        acc = 0
        accs = []
        self.reward = []
        for i in range(self.catnum):
            acc = 0
            correct =0
            predictions, references = [],[]
            for data in self.validationData[i]:
                if isinstance(data, list):  # image classifier
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)
                    outputs = self.net(inputs)
                    predicts = outputs.argmax(dim=1)
                    correct += predicts.eq(labels).sum().item()
                    acc += correct/len(self.validationData[i])
                elif isinstance(data, dict):  # text classifier
                    inputs = {k: v.to(self.device) for k, v in data.items() 
                              if k not in ['labels', 'indices']}
                    labels = data['labels'].to(self.device)
                    outputs = self.net(**inputs)[0]
                    references += labels.tolist()
                    if self.net.num_classes ==1:
                        predictions += outputs.squeeze()
                    else:
                        predictions += outputs.argmax(dim=1).tolist()
                    valid_metric = self.metric.compute(predictions=predictions,references=references)[self.metric_name]
                    acc += valid_metric / len(self.validationData)
                elif isinstance(data,pygBatch):  # graph classifier
                    inputs = data.to(self.device)
                    labels = data.y.to(self.device)
                    outputs = self.net(inputs)
                    predicts = outputs.argmax(dim=1)
                    correct += predicts.eq(labels).sum().item()
                    acc += correct/len(self.validationData[i])
                else:
                    raise NotImplementedError()
            accs.append(acc)

        for i, j in zip(accs, self.accs):
            self.reward.append(i-j)
        self.accs = accs

        for i in range(self.catnum):
            self.total[i] = self.total[i] *(1.0-self.alpha) + self.reward[i] * self.alpha
        # self.total[self.training] = self.total[self.training] * (1.0 - self.alpha) + self.reward * self.alpha
        p = self.policy(np.abs(self.total)if self.abs else self.total)
        temp = np.random.choice(range(self.catnum), p=p)
        data_loader = self._dataloader(self.CLDataset(self.data[temp].dataset)) 
        return data_loader



class RLTeacherNaive(BaseCL):
    def __init__(self, ):
        super(RLTeacherNaive, self).__init__()

        self.name = 'rl_teacher_naive'
        self.policy = EpsilonGreedyPolicy(0.01)

        self.catnum = 10
        self.alpha = 0.1
        self.total = [0 for i in range(self.catnum)]
        self.abs = False

        self.window_size = 10
        self.scores = []
        self.epoch_index = 0
        self.accs = [0 for i in range(self.catnum)]
        self.reward = []


    def data_split(self):
        self.indexs = [range(i*self.data_size//self.catnum, (i+1)*self.data_size//self.catnum) for i in range(self.catnum)]
        l = [len(self.indexs[i])// self.catnum for i in range(self.catnum)]
        self.data = [self._dataloader(Subset(self.dataset, self.indexs[i]))for i in range(self.catnum)]
        self.validationData = [self._dataloader(Subset(self.dataset, self.indexs[i][:l[i]]))for i in range(self.catnum)]


    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)
        self.metric = kwargs.get("metric")
        self.metric_name = kwargs.get("metric_name")
        self.data_split()


    def data_curriculum(self, **kwargs):
        acc = 0
        accs = []
        self.reward = []
        for i in range(self.catnum):
            acc = 0
            correct =0
            predictions, references = [],[]
            for data in self.validationData[i]:
                if isinstance(data, list):  # image classifier
                    inputs= data[0].to(self.device)
                    labels = data[1].to(self.device)
                    outputs = self.net(inputs)
                    _, pred = outputs.max(1)
                    correct = (pred == labels).sum().item()  
                    acc += correct/len(self.validationData[i])
                elif isinstance(data, dict):  # text classifier
                    inputs = {k: v.to(self.device) for k, v in data.items() 
                              if k not in ['labels', 'indices']}
                    labels = data['labels'].to(self.device)
                    outputs = self.net(**inputs)[0]
                    references += labels.tolist()
                    if self.net.num_classes ==1:
                        predictions += outputs.squeeze()
                    else:
                        predictions += outputs.argmax(dim=1).tolist()
                    valid_metric = self.metric.compute(predictions=predictions,references=references)[self.metric_name]
                    acc += valid_metric / len(self.validationData)
                elif isinstance(data,pygBatch):  # graph classifier
                    inputs = data.to(self.device)
                    labels = data.y.to(self.device)
                    outputs = self.net(inputs)
                    predicts = outputs.argmax(dim=1)
                    correct += predicts.eq(labels).sum().item()
                    acc += correct/len(self.validationData[i])
                else:
                    raise NotImplementedError()
            accs.append(acc)
        for i, j in zip(accs, self.accs):
            self.reward.append(i-j)
        self.accs = accs

        self.scores.append(self.accs)
        if self.epoch_index % self.window_size == 0:
            self.reward = estimate_slope(range(len(self.scores)), self.scores)
            self.scores = []
            for i in range(self.catnum):
                self.total[i] = self.total[i] * (1.0 - self.alpha) + self.reward[i] * self.alpha
            p = self.policy(np.abs(self.total)if self.abs else self.total)
            self.training = np.random.choice(range(self.catnum), p=p)
            self.data_loader = self._dataloader(self.CLDataset(self.data[self.training].dataset))
        self.epoch_index += 1

        return self.data_loader

class RLTeacherWindow(BaseCL):
    def __init__(self, ):
        super(RLTeacherWindow, self).__init__()

        self.name = 'rl_teacher_window'
        self.policy = EpsilonGreedyPolicy(0.01)

        # self.partnum = 10
        self.alpha = 0.1
        self.abs = False

        self.acc = 0
        self.training = 0
        self.reward = 0

        self.window_size = 10
        self.epoch_index = 1


    def split(self, data_loader, partnum):
        temp = data_loader.dataset
        k = len(temp)
        l = k // partnum
        self.data = []
        for i in range(partnum-1):
            self.data.append(Subset(temp, range(i * l, (i + 1) * l)))
        self.partnum = partnum - 1
        self.validationData = self._dataloader(Subset(temp, range(self.partnum * l, k)))


    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)
        self.metric = kwargs.get("metric")
        self.metric_name = kwargs.get("metric_name")
        self.split(loader, 10)
        self.total = np.zeros(self.partnum)
        self.scores = [deque(maxlen=self.window_size) for _ in range(self.partnum)]
        self.timesteps = [deque(maxlen=self.window_size) for _ in range(self.partnum)]


    def data_curriculum(self, **kwargs):
        acc = 0
        correct =0
        predictions, references = [],[]
        for data in self.validationData:
            if isinstance(data, list):  # image classifier
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                outputs = self.net(inputs)
                predicts = outputs.argmax(dim=1)
                correct += predicts.eq(labels).sum().item()
                acc += correct/len(self.validationData)
            elif isinstance(data, dict):  # text classifier
                inputs = {k: v.to(self.device) for k, v in data.items() 
                              if k not in ['labels', 'indices']}
                labels = data['labels'].to(self.device)
                outputs = self.net(**inputs)[0]
                references += labels.tolist()
                if self.net.num_classes ==1:
                    predictions += outputs.squeeze()
                else:
                    predictions += outputs.argmax(dim=1).tolist()
                valid_metric = self.metric.compute(predictions=predictions,references=references)[self.metric_name]
                acc += valid_metric / len(self.validationData)
            elif isinstance(data,pygBatch):  # graph classifier
                inputs = data.to(self.device)
                labels = data.y.to(self.device)
                outputs = self.net(inputs)
                predicts = outputs.argmax(dim=1)
                correct += predicts.eq(labels).sum().item()
                acc += correct/len(self.validationData)
            else:
                raise NotImplementedError()
        self.reward, self.acc = acc - self.acc, acc

        self.scores[self.training].append(self.acc)
        self.timesteps[self.training].append(self.epoch_index)
        self.total = [estimate_slope(timesteps, scores) if len(scores) > 1 else 1 for timesteps, scores in
                  zip(self.timesteps, self.scores)]
        p = self.policy(np.abs(self.total) if self.abs else self.total)
        self.training = np.random.choice(range(self.partnum), p=p)
        self.data_loader = self._dataloader(self.CLDataset(self.data[self.training]))
        self.epoch_index += 1
        return self.data_loader



class RLTeacherSampling(BaseCL):
    def __init__(self, ):
        super(RLTeacherSampling, self).__init__()

        self.name = 'rl_teacher_sampling'
        self.policy = EpsilonGreedyPolicy(0.01)

        self.partnum = 10
        self.alpha = 0.1
        window_size = 10
        self.total = [0 for i in range(self.partnum)]
        self.abs = False

        self.accs = []
        self.prevr = np.zeros(self.partnum)
        self.training = 0

        self.window_size = window_size
        self.dscores = deque(maxlen=window_size)
        self.prevr = np.zeros(self.partnum)


    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)
        self.metric = kwargs.get("metric")
        self.metric_name = kwargs.get("metric_name")
        partnum = 10
        temp = self.dataset
        k = len(temp)
        l = k // partnum
        self.data = []
        self.partnum = partnum
        for i in range(partnum-1):
            self.data.append(self._dataloader(Subset(temp, range(i * l, (i + 1) * l))))
        self.data.append(self._dataloader(Subset(temp, range((self.partnum - 1) * l, k))))


    def data_curriculum(self, **kwargs):
        self.accs = []
        for i in range(self.partnum):
            acc = 0
            correct = 0
            predictions, references = [],[]
            for data in self.data[i]:
                if isinstance(data, list):  # image classifier
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)
                    outputs = self.net(inputs)
                    predicts = outputs.argmax(dim=1)
                    correct += predicts.eq(labels).sum().item()
                    acc += correct/len(self.data[i])
                elif isinstance(data, dict):  # text classifier
                    inputs = {k: v.to(self.device) for k, v in data.items() 
                              if k not in ['labels', 'indices']}
                    labels = data['labels'].to(self.device)
                    outputs = self.net(**inputs)[0]
                    references += labels.tolist()
                    if self.net.num_classes ==1:
                        predictions += outputs.squeeze()
                    else:
                        predictions += outputs.argmax(dim=1).tolist()
                    valid_metric = self.metric.compute(predictions=predictions,references=references)[self.metric_name]
                    acc += valid_metric / len(self.data)
                elif isinstance(data,pygBatch):  # graph classifier
                    inputs = data.to(self.device)
                    labels = data.y.to(self.device)
                    outputs = self.net(inputs)
                    predicts = outputs.argmax(dim=1)
                    correct += predicts.eq(labels).sum().item()
                    acc += correct/len(self.data[i])
                else:
                    raise NotImplementedError()
            self.accs.append(acc)

        if len(self.dscores) > 0:
            if isinstance(self.policy, ThompsonPolicy):
                slopes = [np.random.choice(drs) for drs in np.array(self.dscores).T]
            else:
                slopes = np.mean(self.dscores, axis=0)
        else:
            slopes = np.ones(self.partnum)
        p = self.policy(np.abs(slopes) if self.abs else slopes)
        self.training = np.random.choice(range(self.partnum), p=p)
        data_loader = self._dataloader(self.CLDataset(self.data[self.training].dataset))
        dr = [i-j for i, j in zip(self.accs, self.prevr)]
        self.prevr = self.accs
        self.dscores.append(dr)
        return data_loader



class RLTeacherTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, policy):
        
        cl_dict = {'online': RLTeacherOnline,
                   'naive': RLTeacherNaive,
                   'window': RLTeacherWindow,
                   'sampling': RLTeacherSampling}
        cl = cl_dict[policy]()

        super(RLTeacherTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)
