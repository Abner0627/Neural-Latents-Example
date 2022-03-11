# %%
'''
nn.SmoothL1Loss
nn.HuberLoss
nn.InstanceNorm2d
'''
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import os
from scipy.ndimage import gaussian_filter1d
from nlb_tools.make_tensors import h5_to_dict, save_to_h5
from nlb_tools.evaluation import evaluate
import h5py


# %%
tStart = time.time()
print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


# %%
def _GF(ddd, sig, N, C, log=True):
    TL = []
    for n in range(N):
        CL = []
        spike = ddd[n, :, :]
        for c in range(C):
            sp_c = spike[:, c]
            sp_c_gf = gaussian_filter1d(sp_c.astype(np.float32), sig)
            if log:
                CL.append(np.log(sp_c_gf + 1e-10)[:, np.newaxis])
            else:
                CL.append(sp_c_gf[:, np.newaxis])
        TL.append(np.hstack(CL)[np.newaxis, :, :])
    TL = np.vstack(TL)
    return TL

def _Dict(dataset_name, train, valid, T, C):
    train_rates_heldin = train[:, :T, :C]
    train_rates_heldout = train[:, :T, C:]
    eval_rates_heldin = valid[:, :T, :C]
    eval_rates_heldout = valid[:, :T, C:]
    eval_rates_heldin_forward = valid[:, T:, :C]
    eval_rates_heldout_forward = valid[:, T:, C:]

    output_dict = {
        dataset_name: {
            'train_rates_heldin': train_rates_heldin.astype(np.float64),
            'train_rates_heldout': train_rates_heldout.astype(np.float64),
            'eval_rates_heldin': eval_rates_heldin.astype(np.float64),
            'eval_rates_heldout': eval_rates_heldout.astype(np.float64),
            'eval_rates_heldin_forward': eval_rates_heldin_forward.astype(np.float64),
            'eval_rates_heldout_forward': eval_rates_heldout_forward.astype(np.float64)
        }
    }  
    return output_dict

def _shuffle(spk_hi):
    spk_hi_shf = spk_hi[:, :, torch.randperm(spk_hi.size()[2])]
    return spk_hi_shf

def _GF_pt(spk_pt, sig, N, C, log=True):
    spk_np = spk_pt.data.numpy()
    spk_GF_np = _GF(spk_np, sig, N, C, log)
    return torch.from_numpy(spk_GF_np).type(torch.FloatTensor)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# %%
dataset_dict = {
    '000128': 'mc_maze',
    '000127': 'area2_bump',
    '000130': 'dmfc_rsg',
    '000129': 'mc_rtt',
    '000138': 'mc_maze_large'
}
hT_dict = {
    '000128': 7,
    '000127': 6,
    '000130': 15,
    '000129': 6,
    '000138': 7
}
sP = './inFR'
dpath = './data/h5'
data_list = os.listdir('./data/h5')

for idx in ['000127', '000128', '000129', '000130', '000138']:
    idx = '000128'
    dataset_name = dataset_dict[idx]
    print('\n==========' + dataset_name + '==========')
    for d in data_list:
        if d.split('_')[0]==idx:
            if d.split('_')[1]=='train':
                train_F = d
            elif d.split('_')[1]=='eval':
                valid_F = d
            elif d.split('_')[1]=='test':
                test_F = d   
            elif d.split('_')[1]=='target':
                target_F = d               


    # %%
    train_H5 = h5py.File(os.path.join(dpath, train_F), 'r')
    valid_H5 = h5py.File(os.path.join(dpath, valid_F), 'r')
    test_H5 = h5py.File(os.path.join(dpath, test_F), 'r')
    target_dict = h5_to_dict(h5py.File(os.path.join(dpath, target_F), 'r'))

    train_spikes_heldin  = np.array(train_H5['train_spikes_heldin'])
    train_spikes_heldin_forward  = np.array(train_H5['train_spikes_heldin_forward'])
    train_spikes_heldout  = np.array(train_H5['train_spikes_heldout'])
    train_spikes_heldout_forward  = np.array(train_H5['train_spikes_heldout_forward'])

    eval_spikes_heldin = np.array(valid_H5['eval_spikes_heldin'])
    eval_spikes_heldout = np.array(valid_H5['eval_spikes_heldout'])

    test_spikes_heldin = np.array(test_H5['eval_spikes_heldin'])

    N_tra, T, C = train_spikes_heldin.shape
    N_val = eval_spikes_heldin.shape[0]
    N_tes = test_spikes_heldin.shape[0]
    _, To, Co = train_spikes_heldout_forward.shape


    # %%
    train_fd = np.concatenate([train_spikes_heldin_forward, train_spikes_heldout_forward], axis=-1).astype(np.int64)
    train_bd = np.concatenate([train_spikes_heldin, train_spikes_heldout], axis=-1).astype(np.int64)
    train = np.concatenate([train_fd, train_bd], axis=1)


    # %%
    from model_seq2seq import GRU_AT_AE, GRU_HoFw 
    C1 = C//2
    C2 = C - C1
    out_sz = 128  
    model_AE = GRU_AT_AE(C//2, C2, out_sz)
    model_HOFW = GRU_HoFw(T, C, To, Co, hT_dict[idx])

    model_AE.apply(init_weights)
    model_HOFW.apply(init_weights)

    LOSS_TRA = []
    LOSS_VAL = []

    bz = 64
    lr_init = 1e-3
    lr_end = 1e-6
    Epoch = 1500
    sigma = 5

    train_data = torch.from_numpy(train).type(torch.FloatTensor)
    train_label = torch.from_numpy(train).type(torch.FloatTensor)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bz, shuffle=True)

    valid_data = torch.from_numpy(eval_spikes_heldin).type(torch.FloatTensor)
    test_data = torch.from_numpy(test_spikes_heldin).type(torch.FloatTensor)

    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_data)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=bz, shuffle=False)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_data)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bz, shuffle=False)    
    train_dataloader_V = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bz, shuffle=False)



    # %%
    optim_AE = optim.AdamW(model_AE.parameters(), lr=lr_init)
    optim_HOFW = optim.AdamW(model_HOFW.parameters(), lr=lr_init)

    scheduler_AE = optim.lr_scheduler.StepLR(optim_AE, step_size=50, gamma=0.95)
    scheduler_HOFW = optim.lr_scheduler.StepLR(optim_HOFW, step_size=50, gamma=0.95)

    lossRect = nn.HuberLoss()
    lossEmbe = nn.HuberLoss()
    lossHoFw = nn.HuberLoss()

    model_AE = model_AE.to(device)
    model_HOFW = model_HOFW.to(device)
    lossRect = lossRect.to(device)
    lossEmbe = lossEmbe.to(device)
    lossHoFw = lossHoFw.to(device)

    # %%
    print('\n------Training------')
    for epoch in range(Epoch):
        model_AE.train()
        model_HOFW.train()
        for n, (Data, Label) in enumerate(train_dataloader):
            # process
            N_pt = Data.size(0)
            spk_data = _shuffle(Data)
            spk_data_hi = _GF_pt(spk_data[:, :T, :C], sigma, N_pt, C)
            spk_data_hi_1 = spk_data_hi[:, :, :C1]
            spk_data_hi_2 = spk_data_hi[:, :, C1:]

            # spk_data_LB = _GF_pt(spk_data, sigma, N_pt, C+Co)
            spk_data_LB = _GF_pt(spk_data, sigma, N_pt, C+Co, log=False)

            spk_data_hi_1 = spk_data_hi_1.to(device)
            spk_data_hi_2 = spk_data_hi_2.to(device)
            spk_data_LB = spk_data_LB.to(device)
            # Rect + HOFW
            x1_enc, x2_enc, x1_rec, x2_rec = model_AE(spk_data_hi_1, spk_data_hi_2)
            x_rec = torch.cat([x1_rec, x2_rec], axis=-1)
            x_rec_TRSP = x_rec.transpose(1, 2)
            x_rec_ho, x_rec_fw = model_HOFW(x_rec, x_rec_TRSP)

            x_rec_ho_TRSP = x_rec_ho.transpose(1, 2)
            x_rec_fw_ho, x_rec_ho_fw = model_HOFW(x_rec_fw, x_rec_ho_TRSP)

            FR_fd = torch.cat([x_rec_fw, x_rec_ho_fw], axis=-1)
            FR_bd = torch.cat([x_rec, x_rec_ho], axis=-1)
            FR = torch.cat([FR_fd, FR_bd], axis=1)            
            # update
            Rloss = lossRect(FR, spk_data_LB)
            Eloss = lossEmbe(x1_enc, x2_enc)
            HFloss = lossHoFw(x_rec_fw_ho, x_rec_ho_fw)
            loss = Rloss + Eloss + HFloss

            optim_AE.zero_grad()
            optim_HOFW.zero_grad()
            loss.backward()
            optim_AE.step()
            optim_HOFW.step()

        print('epoch[{}], Train loss:{:.4f}, Rloss:{:.4f}, Eloss:{:.4f}, HFloss:{:.4f}'\
            .format(epoch+1, loss.item(), Rloss.item(), Eloss.item(), HFloss.item())) 
        LOSS_TRA.append(
            {'tot': loss.item(), 
            'rect': Rloss.item(),
            'emb': Eloss.item(),
            'hf': HFloss.item()}
        )     
        if loss.item() < 0.01:
            print('traning loss convg')
            break       
        
    print('\n------Validation------')
    model_AE.eval()
    model_HOFW.eval()
    with torch.no_grad():
        VAL_L, TRA_L = [], []
        # ==================================
        for n, (Val_Data, Tes_Data) in enumerate(valid_dataloader):
            # process
            n_val = Val_Data.size(0)
            Val_Data_GF = _GF_pt(Val_Data, sigma, n_val, C)
            spk_data_hi_1_val = Val_Data_GF[:, :, :C1]
            spk_data_hi_2_val = Val_Data_GF[:, :, C1:] 

            spk_data_hi_1_val = spk_data_hi_1_val.to(device)
            spk_data_hi_2_val = spk_data_hi_2_val.to(device)
            
            _, _, x1_val_rec, x2_val_rec = model_AE(spk_data_hi_1_val, spk_data_hi_2_val)
            x_val_rec = torch.cat([x1_val_rec, x2_val_rec], axis=-1)
            x_val_rec_TRSP = x_val_rec.transpose(1, 2)
            x_val_rec_ho, x_val_rec_fw = model_HOFW(x_val_rec, x_val_rec_TRSP)

            x_val_rec_ho_TRSP = x_val_rec_ho.transpose(1, 2)
            x_val_rec_fw_ho, x_val_rec_ho_fw = model_HOFW(x_val_rec_fw, x_val_rec_ho_TRSP)

            FR_val_fd = torch.cat([x_val_rec_fw, x_val_rec_ho_fw], axis=-1)
            FR_val_bd = torch.cat([x_val_rec, x_val_rec_ho], axis=-1)
            FR_val = torch.cat([FR_val_fd, FR_val_bd], axis=1)

            VAL_L.append(FR_val.cpu().data.numpy())
        valid_inferred = np.vstack(VAL_L)
        # ==================================
        # ==================================
        for n, (Data_V, Label_V) in enumerate(train_dataloader_V):
            # process
            n_tra = Data_V.size(0)
            Data_V_hi = Data_V[:, :T, :C]
            Data_V_hi_GF = _GF_pt(Data_V_hi, sigma, n_tra, C)  
            spk_data_hi_1_tra = Data_V_hi_GF[:, :, :C1]
            spk_data_hi_2_tra = Data_V_hi_GF[:, :, C1:] 

            spk_data_hi_1_tra = spk_data_hi_1_tra.to(device)
            spk_data_hi_2_tra = spk_data_hi_2_tra.to(device)
            
            _, _, x1_tra_rec, x2_tra_rec = model_AE(spk_data_hi_1_tra, spk_data_hi_2_tra)
            x_tra_rec = torch.cat([x1_tra_rec, x2_tra_rec], axis=-1)
            x_tra_rec_TRSP = x_tra_rec.transpose(1, 2)
            x_tra_rec_ho, x_tra_rec_fw = model_HOFW(x_tra_rec, x_tra_rec_TRSP)

            x_tra_rec_ho_TRSP = x_tra_rec_ho.transpose(1, 2)
            x_tra_rec_fw_ho, x_tra_rec_ho_fw = model_HOFW(x_tra_rec_fw, x_tra_rec_ho_TRSP)

            FR_tra_fd = torch.cat([x_tra_rec_fw, x_tra_rec_ho_fw], axis=-1)
            FR_tra_bd = torch.cat([x_tra_rec, x_tra_rec_ho], axis=-1)
            FR_tra = torch.cat([FR_tra_fd, FR_tra_bd], axis=1)

            TRA_L.append(FR_tra.cpu().data.numpy())
        train_inferred = np.vstack(TRA_L)
        # ==================================        
        output_dict = _Dict(dataset_name, train_inferred, valid_inferred, T, C)
        EVA = evaluate(target_dict, output_dict)

            
        print('evaluate:{}'.format(EVA))
        LOSS_VAL.append(EVA)   
            



    # %%
    print('\n------Testing------')
    model_AE.eval()
    model_HOFW.eval()
    with torch.no_grad():
        TES_L = []
        # ==================================
        for n, (Tes_Data, Val_Data) in enumerate(test_dataloader):
            # process
            n_tes = Tes_Data.size(0)
            Tes_Data_GF = _GF_pt(Tes_Data, sigma, n_tes, C)        
            spk_data_hi_1_tes = Tes_Data_GF[:, :, :C1]
            spk_data_hi_2_tes = Tes_Data_GF[:, :, C1:] 

            spk_data_hi_1_tes = spk_data_hi_1_tes.to(device)
            spk_data_hi_2_tes = spk_data_hi_2_tes.to(device)
            
            _, _, x1_tes_rec, x2_tes_rec = model_AE(spk_data_hi_1_tes, spk_data_hi_2_tes)
            x_tes_rec = torch.cat([x1_tes_rec, x2_tes_rec], axis=-1)
            x_tes_rec_TRSP = x_tes_rec.transpose(1, 2)
            x_tes_rec_ho, x_tes_rec_fw = model_HOFW(x_tes_rec, x_tes_rec_TRSP)

            x_tes_rec_ho_TRSP = x_tes_rec_ho.transpose(1, 2)
            x_tes_rec_fw_ho, x_tes_rec_ho_fw = model_HOFW(x_tes_rec_fw, x_tes_rec_ho_TRSP)

            FR_tes_fd = torch.cat([x_tes_rec_fw, x_tes_rec_ho_fw], axis=-1)
            FR_tes_bd = torch.cat([x_tes_rec, x_tes_rec_ho], axis=-1)
            FR_tes = torch.cat([FR_tes_fd, FR_tes_bd], axis=1)

            TES_L.append(FR_tes.cpu().data.numpy())
        test_inferred = np.vstack(TES_L)
        # ==================================


    # %%
    print('\n------Saving------')
    output_dict = _Dict(dataset_name, np.vstack([train_inferred, valid_inferred]), test_inferred, T, C)

    for k in output_dict[dataset_name].keys():
        print(k + ': [{}]'.format(output_dict[dataset_name][k].shape))

    save_to_h5(output_dict, './inFR/finnal_SML1seq.h5', overwrite=True)

    tra_npy_fn = dataset_name + '_train_loss_SML1seq.npy'
    val_npy_fn = dataset_name + '_valid_loss_SML1seq.npy'
    np.save(os.path.join('./inFR', tra_npy_fn), LOSS_TRA)
    np.save(os.path.join('./inFR', val_npy_fn), LOSS_VAL)

# %%
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))





