# %%
import h5py 
import numpy as np
import os
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

# %%
dataset_dict = {
    '000128': 'mc_maze',
    '000127': 'area2_bump',
    '000130': 'dmfc_rsg',
    '000129': 'mc_rtt',
    '000138': 'mc_maze_large'
}
data_list = os.listdir('./data')
sP = './data/h5'
for idx in ['000128', '000127', '000130', '000129', '000138']:
# idx = '000128'
    dataset_name = dataset_dict[idx]
    print('==========' + dataset_name + '==========')
    for d in data_list:
        if d.split('_')[0]==idx:
            if d.split('_')[2]=='desc-test':
                fn_tes = os.path.join('./data', d)
            elif d.split('_')[2]=='desc-train':
                fn = os.path.join('./data', d)


    # %%
    bin_width = 5
    dataset = NWBDataset(fn)
    dataset_tes = NWBDataset(fn_tes)
    dataset.resample(bin_width)
    dataset_tes.resample(bin_width)


    # %%
    # ## Make train input data
    # # Generate input tensors
    train_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split='train', save_path=os.path.join(sP, idx+"_train_input.h5"), save_file=True, include_forward_pred=True)
    # ## Make eval input data
    # # Generate input tensors
    eval_dict = make_eval_input_tensors(dataset, dataset_name=dataset_name, trial_split='val', save_path=os.path.join(sP, idx+"_eval_input.h5"), save_file=True)
    # ## Make test input data
    # # Generate input tensors
    test_dict = make_eval_input_tensors(dataset_tes, dataset_name=dataset_name, trial_split='test', save_path=os.path.join(sP, idx+"_test_input.h5"), save_file=True)

    # %%
    target_dict = make_eval_target_tensors(dataset, dataset_name, 'train', 'val', save_file=True, save_path=os.path.join(sP, idx+"_target_input.h5"), include_psth=True)
