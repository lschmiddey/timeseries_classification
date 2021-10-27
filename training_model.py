#%%
from common import *
from config import *
from dataloaders import *
from transform_data import *
from model_part import *
from callbacks import *

from functools import partial
from scipy.io import arff

# %%
# Read data
data_train = arff.loadarff('data/LargeKitchenAppliances_TRAIN.arff')
data_test = arff.loadarff('data/LargeKitchenAppliances_TEST.arff')
df_test = pd.DataFrame(data_test[0])
df_train = pd.DataFrame(data_train[0])

# let's add a categorical variable
countries = ['Germany', 'US']
df_train["country"] = np.random.choice(countries, len(df_train))
df_test["country"] = np.random.choice(countries, len(df_test))

df_train.head()
# %%
# transform data
x, emb_vars, y, info_vars, cat_dict, cat_inv_dict = transform_data(df, model_name)


# save dictionaries
save_obj(cat_dict, f'{MODEL_PATH}/{model_name}_cat_dict')
save_obj(cat_inv_dict, f'{MODEL_PATH}/{model_name}_cat_inv_dict')
#%%
# put data into training/test/
device = DEVICE
datasets = create_datasets(x, emb_vars, y, info_vars, valid_pct=0.2, test_pct=0.3, seed=1234)
data = DataBunch(*create_loaders(datasets, bs=1024))
# %%
# define model
raw_feat = x.shape[1]
emb_dims = [(len(cat_dict[0]), GAME_EMB_DIMS), (len(cat_dict[1]), MARKET_EMB_DIMS)]

num_classes = 2

# create model
model = Classifier_3days(raw_feat, emb_dims, num_classes).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

# %%
lr_learn = Learner(model, opt, loss_func, data)
lr_run = Runner(cb_funcs=[LR_Find, Recorder])
lr_run.fit(2, lr_learn)
lr_run.recorder.plot(skip_last=5)
#%%
model = Classifier_3days(raw_feat, emb_dims, num_classes).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)
cbfs = [Recorder, partial(AvgStatsCallback,accuracy)]
learn = Learner(model, opt, loss_func, data)
run = Runner(cb_funcs=cbfs)
run.fit(10, learn)
#%%

def append_stats(i, mod, inp, outp):
    act_means[i].append(outp.data.mean())
    act_stds [i].append(outp.data.std())


model = Classifier_3days(raw_feat, emb_dims, num_classes).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)
cbfs = [Recorder, partial(AvgStatsCallback,accuracy)]
learn = Learner(model, opt, loss_func, data)
run = Runner(cb_funcs=cbfs)

act_means = [[] for _ in model.raw]
act_stds  = [[] for _ in model.raw]

for i,m in enumerate(model.raw): m.register_forward_hook(partial(append_stats, i))

run.fit(10, learn)
# %%
plt.plot(act_means[0])

# %%
plt.plot(act_stds[0])
#%%
### Now with leaky ReLU and Kaiming init
### Does not seem to work at all

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        torch.nn.init.kaiming_normal_
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.kaiming_normal_
        torch.nn.init.zeros_(m.bias)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv1d') != -1:
#         torch.nn.init.normal_(m.weight, 0.0, 0.02)
#     elif classname.find('BatchNorm1d') != -1:
#         torch.nn.init.normal_(m.weight, 1.0, 0.02)
#         torch.nn.init.zeros_(m.bias)

opt = optim.Adam(model.parameters(), lr=0.01)

model = Classifier_3days(raw_feat, emb_dims, num_classes).to(device)

model.apply(weights_init)

act_means = [[] for _ in model.raw]
act_stds  = [[] for _ in model.raw]

for i,m in enumerate(model.raw): m.register_forward_hook(partial(append_stats, i))

learn = Learner(model, opt, loss_func, data)
run = Runner(cb_funcs=cbfs)
run.fit(10, learn)
# %%
plt.plot(act_means[0])

# %%
plt.plot(act_stds[0])
# %%