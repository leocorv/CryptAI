#!/usr/bin/env python3
"""One script to train them all — LSTM, Transformer, TCN, MLP. Batched validation, GPU max."""
import os,sys,time,glob,json
import numpy as np
import pandas as pd
import torch

BASE="/mnt/hive_storage/CryptAI"
sys.path.insert(0,f"{BASE}/lib")
from features import build_features
os.makedirs(f"{BASE}/training/checkpoints",exist_ok=True)

model_type=sys.argv[1] if len(sys.argv)>1 else "transformer"
print(f"[train] Model: {model_type}",flush=True)

# Configs
CONFIGS={
    "transformer":{"bs":1024,"lr":1e-4},
    "tcn":{"bs":8192,"lr":1e-3},
    "lstm":{"bs":512,"lr":1e-3},
    "mlp":{"bs":2048,"lr":1e-3}
}
cfg=CONFIGS.get(model_type,CONFIGS["transformer"])
BATCH=cfg["bs"]
LR=cfg["lr"]

# Load data
print("[data] Loading...",flush=True)
all_X,all_y=[],[]
for f in sorted(glob.glob(f"{BASE}/data/*/*/historical.csv")):
    try:
        df=pd.read_csv(f,parse_dates=["timestamp"])
        if len(df)<100:continue
        feats=build_features(df)
        if len(feats)<96:continue
        close=df["close"].values[-len(feats):]
        ret=np.diff(close,prepend=close[0])/(close+1e-8)
        labels=np.where(ret<-0.002,0,np.where(ret>0.002,2,1))
        vals=feats.values.astype(np.float32)
        for i in range(len(vals)):
            m=vals[i].mean();s=vals[i].std()+1e-8;vals[i]=(vals[i]-m)/s
        for i in range(len(vals)-96):
            all_X.append(vals[i:i+96]);all_y.append(int(labels[i+96]))
    except:pass

X=np.array(all_X);y=np.array(all_y);del all_X,all_y
n=len(X);perm=np.random.permutation(n)
s=int(n*0.7);vs=int(n*0.85)
Xt,Xv,Xte=X[perm[:s]],X[perm[s:vs]],X[perm[vs:]]
yt,yv,yte=y[perm[:s]],y[perm[s:vs]],y[perm[vs:]]
print(f"  Train:{len(Xt)} Val:{len(Xv)} Test:{len(Xte)}",flush=True)

# Model
input_dim=Xt.shape[2]
if model_type=="transformer":
    from models_v2 import DeepTransformer as M; m=M(input_dim,256,8,6,96).to("cuda")
elif model_type=="tcn":
    from models_v2 import TCN as M; m=M(input_dim,256,6,96).to("cuda")
elif model_type=="lstm":
    from models_v2 import DeepLSTM as M; m=M(input_dim,256,2,96).to("cuda")
else:
    from models import MLPDirection as M; m=M(input_dim*96).to("cuda")

print(f"  Params: {sum(p.numel() for p in m.parameters()):,}",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5)
crit=torch.nn.CrossEntropyLoss()

best,pat,t0=0,0,time.time()
for ep in range(100):
    p=torch.randperm(len(Xt))
    for i in range(0,len(p),BATCH):
        idx=p[i:i+BATCH]
        xb=torch.from_numpy(Xt[idx]).to("cuda")
        yb=torch.from_numpy(yt[idx]).to("cuda").long()
        opt.zero_grad(set_to_none=True)
        crit(m(xb),yb).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        opt.step()
    
    # Batched validation
    m.eval();vl_tot=acc_tot=n_b=0
    with torch.no_grad():
        for j in range(0,len(Xv),2048):
            xv=torch.from_numpy(Xv[j:j+2048]).to("cuda")
            yv_=torch.from_numpy(yv[j:j+2048]).to("cuda").long()
            o=m(xv)
            vl_tot+=crit(o,yv_).item()*len(xv)
            acc_tot+=(o.argmax(1)==yv_).float().sum().item()
            n_b+=len(xv)
    vl=vl_tot/n_b;acc=acc_tot/n_b;m.train();sched.step(vl)
    
    if acc>best:best=acc;pat=0;torch.save({"state":m.state_dict(),"val_acc":acc},f"{BASE}/training/checkpoints/{model_type}_best.pth")
    else:pat+=1
    vram=torch.cuda.memory_allocated(0)/1024/1024/1024
    print(f"{model_type.upper()} EP{ep:>3} | VL:{vl:.4f} | Acc:{acc:.3f} | Best:{best:.3f} | VRAM:{vram:.2f}GB | {time.time()-t0:.0f}s",flush=True)
    if pat>=15:print(f"Early stop EP{ep}",flush=True);break
print(f"Done: Best={best:.3f}",flush=True)