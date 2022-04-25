# %% Packages
import torch
import torch.nn as nn

# %%
class Temp_AT(nn.Module):
    def __init__(self, fdim):
        super (Temp_AT, self).__init__()
        self.t_at = nn.Sequential(
            nn.Linear(fdim, fdim//4),
            nn.LayerNorm(fdim//4),
            nn.Tanh(),
            nn.Linear(fdim//4, 1),
            nn.Flatten(),
            nn.Softmax(-1)
            )
    def forward(self, x):
        # IN bz,tap,in_sz
        at_map = self.t_at(x).unsqueeze(-1)
        y = at_map * x
        # y = torch.sum(y, dim=1, keepdim=True)
        return y, at_map

class MAT(nn.Module):
    def __init__(self, nt, heads, dropout):
        super(MAT, self).__init__()
        self.linear_k = nn.Linear(nt, nt)
        self.linear_v = nn.Linear(nt, nt)
        self.linear_q = nn.Linear(nt, nt)
        
        self.dim_per_head = nt // heads
        self.heads = heads
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.LN = nn.LayerNorm(nt)
    def forward(self, x):
        dim_per_head = self.dim_per_head
        heads = self.heads
        nt = dim_per_head * heads
        batch_size = x.size(0)
        # linear projection
        key = self.linear_k(x)
        value = self.linear_v(x)
        query = self.linear_q(x)
        # split by heads
        key = key.view(batch_size * heads, -1, dim_per_head)
        value = value.view(batch_size * heads, -1, dim_per_head)
        query = query.view(batch_size * heads, -1, dim_per_head)   
        # self-attention
        attention = torch.bmm(query, key.transpose(1, 2))    # Q . K^T
        nt_at = attention.size(-1)
        scale = (nt_at) ** -0.5        
        attention = attention * scale    # (Q . K^T) / sqrt(d)
        
        attention = self.softmax(nn.functional.relu(attention)) 
        attention_fin = self.dropout(attention)
        # ------------------------
        attention_out = torch.sum(attention_fin, dim=1)
        at_sum_sum = attention_out.view(batch_size, heads, -1)
        # (bz, heads, nt)        
        
        context = torch.bmm(attention_fin, value)
        output = context.view(batch_size,-1, nt)
        out_ = self.LN(x + output)
        return out_, at_sum_sum

class GRU_AT_LN(nn.Module):
    def __init__(self, in_sz, out_sz, hid=2, bid=True, LN=True, AT=True):
        super (GRU_AT_LN, self).__init__()
        self.hid = hid
        self.LN = LN
        self.AT = AT
        #====
        if hid==2:
            self.GRU_D1 = nn.GRU(in_sz, in_sz, 1, batch_first=True, bidirectional=bid)
            if bid:
                in2_sz = int(in_sz*2)
            else:
                in2_sz = in_sz
            self.GRU_D2 = nn.GRU(in2_sz, out_sz, 1, batch_first=True, bidirectional=bid)
        else:
            self.GRU_S = nn.GRU(in_sz, out_sz, 1, batch_first=True, bidirectional=bid)
        #====
        if AT:
            if bid:
                self.temp_at = Temp_AT(int(out_sz*2))
            else:
                self.temp_at = Temp_AT(out_sz)
           
    def forward(self, x):
        # IN bz,tap,in_sz
        bz = x.size(0)
        # GRU
        if self.hid==2:
            x, _ = self.GRU_D1(x)
            if self.LN:
                x = nn.functional.layer_norm(x, x.size()[-1:])
            x, _ = self.GRU_D2(x)
            if self.LN:
                x = nn.functional.layer_norm(x, x.size()[-1:])
        else:
            x, _ = self.GRU_S(x)
            if self.LN:
                x = nn.functional.layer_norm(x, x.size()[-1:])
        # AT
        if self.AT:
            x, at_map = self.temp_at(x)
        return x

# class GRU_AT(nn.Module):
#     def __init__(self, in_sz, out_sz, numlayer=2, bid=True):
#         super(GRU_AT, self).__init__()
#         self.GRU_layer = nn.GRU(in_sz, out_sz, numlayer, batch_first=True, bidirectional=bid)
#         if bid:
#             self.AT = Temp_AT(int(out_sz*2))
#         else:
#             self.AT = Temp_AT(out_sz)
#     def forward(self, x):
#         pred, _ = self.GRU_layer(x)
#         pred_AT, _ = self.AT(pred)
#         return pred_AT

# %%
class GRU_AT_AE(nn.Module):
    def __init__(self, C1, C2, out_sz):
        super(GRU_AT_AE, self).__init__()
        self.GRU_AT_layer1_enc = GRU_AT_LN(C1, out_sz)
        self.GRU_AT_layer2_enc = GRU_AT_LN(C2, out_sz)
        self.GRU_AT_layer1_dec = GRU_AT_LN(out_sz*2, C1, bid=False, AT=False)
        self.GRU_AT_layer2_dec = GRU_AT_LN(out_sz*2, C2, bid=False, AT=False) 
    def forward(self, x1, x2):
        x1_enc = self.GRU_AT_layer1_enc(x1)
        x2_enc = self.GRU_AT_layer2_enc(x2)
        x1_rec = self.GRU_AT_layer1_dec(x1_enc)
        x2_rec = self.GRU_AT_layer2_dec(x2_enc)
        return x1_enc, x2_enc, nn.functional.relu(x1_rec), nn.functional.relu(x2_rec)

class GRU_HoFw(nn.Module):
    def __init__(self, T, C, To, Co, hT):
        super(GRU_HoFw, self).__init__()
        self.GRU_AT_layer_ho = GRU_AT_LN(C, Co, bid=False)
        # self.MAT_fw = MAT(T, hT, 0.1)
        self.MLP_fw = nn.Sequential(
            nn.Linear(T, To),
            nn.ReLU()
            # nn.Linear(64, To),
            # nn.ReLU()
        )
    def forward(self, x_rec, x_rec_TRSP):
        # bz, T, C = x_rec.size()
        x_rec_ho = self.GRU_AT_layer_ho(x_rec)
        # x_rec_fw_mid, _ = self.MAT_fw(x_rec_TRSP)
        x_rec_fw = self.MLP_fw(x_rec_TRSP).transpose(1, 2)
        return nn.functional.relu(x_rec_ho), x_rec_fw

# %% Test
if __name__ == "__main__":
    C, out_sz = 137, 128
    C2 = C - C//2
    T, To, Co = 140, 40, 45
    hT = 7
    F = GRU_AT_AE(C//2, C2, out_sz)
    F2 = GRU_HoFw(T, C, To, Co, hT)

    x1 = torch.rand(32, T, C//2)
    x2 = torch.rand(32, T, C2)
    x1_enc, x2_enc, x1_rec, x2_rec = F(x1, x2)
    x_rec = torch.cat([x1_rec, x2_rec], axis=-1)
    x_rec_TRSP = x_rec.transpose(1,2)
    x_rec_ho, x_rec_fw = F2(x_rec, x_rec_TRSP)

    print(x1_enc.size())
    print(x2_enc.size())
    print(x1_rec.size())
    print(x2_rec.size())    
    print(x_rec_ho.size()) 
    print(x_rec_fw.size()) 