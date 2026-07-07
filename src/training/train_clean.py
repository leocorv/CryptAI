#!/usr/bin/env python3
"""CryptAI Clean Trainer v3 — z-score labels, balanced per timeframe, temporal split"""
import os,sys,time,glob,json
import numpy as np
import pandas as pd
import torch

BASE="/mnt/hive_storage/CryptAI"
sys.path.insert(0,f"{BASE}/lib")
from features import build_features
os.makedirs(f"{BASE}/training/checkpoints",exist_ok=True)

model_type=sys.argv[1] if len(sys.argv)>1 else "tcn"
CONFIGS={"tcn":{"bs":2048,"lr":1e-3},"transformer":{"bs":512,"lr":1e-4},"lstm":{"bs":256,"lr":1e-3}}
cfg=CONFIGS.get(model_type,CONFIGS["tcn"])
BATCH=cfg["bs"];LR=cfg["lr"]

print(f"[train] Model: {model_type} | Z-score labels per timeframe",flush=True)

all_X,all_y,all_ts=[],[],[]
for f in sorted(glob.glob(f"{BASE}/data/*/*/historical.csv")):
    try:
        parts=f.split("/");sym,tf=parts[-3],parts[-2]
        df=pd.read_csv(f,parse_dates=["timestamp"])
        if len(df)<200:continue
        feats=build_features(df)
        if len(feats)<196:continue
        close=df["close"].values[-len(feats):]
        vals=feats.values.astype(np.float32);ts=df["timestamp"].values[-len(feats):]
        
        # Z-score labels: label buy/sell if return exceeds 1 std of recent returns
        ret=close[1:]/close[:-1]-1
        # Rolling z-score
        window=min(100,len(ret))
        mean_r=np.convolve(ret,np.ones(window)/window,mode='same')
        std_r=np.sqrt(np.convolve(ret**2,np.ones(window)/window,mode='same')-mean_r**2)+1e-8
        z_scores=(ret-mean_r)/std_r
        labels=np.where(z_scores<-1.0,0,np.where(z_scores>1.0,2,1))
        
        for i in range(len(vals)-96):
            all_X.append(vals[i:i+96]);all_y.append(int(labels[i+96]));all_ts.append(ts[i+96])
    except:pass

X=np.array(all_X);y=np.array(all_y);ts=np.array(all_ts)
del all_X,all_y,all_ts

# Temporal split
order=np.argsort(ts);X=X[order];y=y[order];ts=ts[order];del order
n=len(X);s=int(n*0.7);vs=int(n*0.85)
X_train,X_val,X_test=X[:s],X[s:vs],X[vs:]
y_train,y_val,y_test=y[:s],y[s:vs],y[vs:]

# Balance check
for name,yd in [("train",y_train),("val",y_val),("test",y_test)]:
    c=pd.Series(yd).value_counts(normalize=True).sort_index()
    print(f"  {name}: {c.to_dict()}",flush=True)

print(f"  Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}",flush=True)

# Norm on train
mean=X_train.mean(axis=(0,1),keepdims=True)
std=X_train.std(axis=(0,1),keepdims=True)+1e-8
X_train=(X_train-mean)/std;X_val=(X_val-mean)/std;X_test=(X_test-mean)/std

# Compute class weights (inverse frequency)
cw=np.bincount(y_train)
cw=1.0/(cw.astype(float)+1e-8);cw=cw/cw.sum()*len(cw)
print(f"  Class weights: {cw.round(3).tolist()}",flush=True)

# Model
input_dim=X_train.shape[2]
if model_type=="transformer":
    from models_v2 import DeepTransformer as M;m=M(input_dim,128,4,4,96).to("cuda")
elif model_type=="tcn":
    from models_v2 import TCN as M;m=M(input_dim,128,4,96).to("cuda")
elif model_type=="lstm":
    from models_v2 import DeepLSTM as M;m=M(input_dim,128,2,96).to("cuda")
else:
    from models import MLPDirection as M;m=M(input_dim*96).to("cuda")

print(f"  Params: {sum(p.numel() for p in m.parameters()):,}",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5)
crit=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to("cuda"))

best,pat,t0=0,0,time.time()
for ep in range(50):
    for i in range(0,len(X_train),BATCH):
        idx=slice(i,min(i+BATCH,len(X_train)))
        xb=torch.from_numpy(X_train[idx]).to("cuda")
        yb=torch.from_numpy(y_train[idx]).to("cuda").long()
        opt.zero_grad(set_to_none=True);crit(m(xb),yb).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    
    m.eval()
    metrics={}
    for name,Xd,yd in [("train",X_train,y_train),("val",X_val,y_val),("test",X_test,y_test)]:
        vl_tot=acc_tot=nb=0
        with torch.no_grad():
            for j in range(0,len(Xd),2048):
                xv=torch.from_numpy(Xd[j:j+2048]).to("cuda")
                yv_=torch.from_numpy(yd[j:j+2048]).to("cuda").long()
                o=m(xv)
                vl_tot+=crit(o,yv_).item()*len(xv)
                acc_tot+=(o.argmax(1)==yv_).float().sum().item();nb+=len(xv)
        metrics[name]=(vl_tot/nb,acc_tot/nb)
    m.train();sched.step(metrics["val"][0])
    
    vl_acc=metrics["val"][1];t_acc=metrics["test"][1];tr_acc=metrics["train"][1]
    if vl_acc>best:best=vl_acc;pat=0;torch.save({"state":m.state_dict(),"val_acc":vl_acc,"test_acc":t_acc},f"{BASE}/training/checkpoints/{model_type}_clean.pth")
    else:pat+=1
    vram=torch.cuda.memory_allocated(0)/1024/1024/1024
    print(f"{model_type.upper()} EP{ep:>3} | TrAcc:{tr_acc:.3f} VlAcc:{vl_acc:.3f} TeAcc:{t_acc:.3f} | Best:{best:.3f} | VRAM:{vram:.2f}GB | {time.time()-t0:.0f}s",flush=True)
    if pat>=10:print(f"Early stop EP{ep}",flush=True);break

print(f"Done: Val={best:.3f} Test={metrics['test'][1]:.3f}",flush=True)
if best>0.70:print(f"WARNING: Acc {best:.3f} > 0.70",flush=True)
elif best>0.55:print(f"Good: Acc {best:.3f}",flush=True)
else:print(f"Clean: Acc {best:.3f}",flush=True)