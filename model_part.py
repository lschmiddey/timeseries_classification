from config import *
from callbacks import *
from common import *

from torch.nn import functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data


class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        self.in_train = False
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property
    def opt(self):       return self.learn.opt
    @property
    def model(self):     return self.learn.model
    @property
    def loss_func(self): return self.learn.loss_func
    @property
    def data(self):      return self.learn.data

    def one_batch(self, xb, emb, yb):
        try:
            self.xb,self.emb,self.yb = xb,emb,yb
            self('begin_batch')
            self.pred = self.model(self.xb,self.emb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,emb,yb in dl: self.one_batch(xb, emb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def fit(self, epochs, learn):
        self.epochs,self.learn,self.loss = epochs,learn,torch.tensor(0.)

        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None

    def predict(self, learn, test_dl):
        self.learn = learn
        self.test_dl = test_dl
        self.preds_array = np.array([]).astype(int)
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_dl:
                xb,emb,_ = batch
                out = self.model(xb,emb)
                preds = F.log_softmax(out, dim=1).argmax(dim=1).numpy()
                self.preds_array = np.concatenate((self.preds_array, preds), axis=None)
        return self.preds_array

    def predict_metrics(self, learn, test_dl, labels):
        self.learn = learn
        self.test_dl = test_dl
        self.labels = labels
        self.preds_array = np.array([]).astype(int)
        self.y_test = np.array([]).astype(int)
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_dl:
                xb,emb,yb = batch
                out = self.model(xb,emb)
                preds = F.log_softmax(out, dim=1).argmax(dim=1).numpy()
                self.preds_array = np.concatenate((self.preds_array, preds), axis=None)
                self.y_test = np.concatenate((self.y_test, yb), axis=None)
        cm = confusion_matrix(self.y_test, self.preds_array)
        
        return plot_confusion_matrix(cm, target_names=self.labels)

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
        return res


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)

def conv1d(ni, nf, ks, stride):
    return nn.Sequential(
        nn.Conv1d(ni, nf, ks, stride=stride, padding=0), nn.BatchNorm1d(nf), nn.ReLU())


def get_cnn_layers(input_shape, output_shapes:list, kernels:list, strides:list, drop=.5):
    output_shapes = [input_shape] + output_shapes
    return [
        conv1d(output_shapes[i], output_shapes[i+1], kernels[i], strides[i])
        for i in range(len(output_shapes)-1)
    ] + [nn.MaxPool1d(2, stride=4), Flatten(), nn.Dropout(drop), nn.Linear(output_shapes[-1], 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(64, 64), nn.ReLU(inplace=True)]


class Classifier_CNN(nn.Module):
    """Model Baseclass."""

    def __init__(self, conv_layers, emb_dims, no):
        super().__init__()

        self.raw = conv_layers

        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.emb_dims = emb_dims

        self.emb_out = nn.Sequential(
            nn.Linear(no_of_embs, 64), nn.ReLU(inplace=True), nn.Linear(64, 64))

        self.out = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

    def forward(self, t_raw, embeddings):  # this is where the data flows in later in the training
        raw_out = self.raw(t_raw)
        emb = [emb_layer(embeddings[:, i].long()) for i, emb_layer in enumerate(self.embeddings)]
        # we want to concatenate convolutions and embeddings. Embeddings are of size (batch_size, no_of_embs),
        # convolution of size (batch, 256, 1) so we need to add another dimension to the embeddings at dimension 2 (
        # counting starts from 0)
        emb_cat = torch.cat(emb, 1)
        emb_cat = self.emb_out(emb_cat)
        t_in = torch.cat([raw_out, emb_cat], dim=1)
        out = self.out(t_in)
        return out


def accuracy(out, yb):
    return (torch.argmax(out, dim=1)==yb).float().mean()

def adjusted_accu(out, yb):
    return (F.log_softmax(out, dim=1).argmax(dim=1)==yb).float().mean()