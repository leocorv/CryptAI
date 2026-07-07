#!/usr/bin/env python3
"""TCN — batch 2048, depth 6, kernel 5"""
import os,sys,time,glob,numpy as np,pandas as pd,torch
BASE="/mnt/hive_storage/CryptAI"
sys.path.insert(0,f"{BASE}/lib");sys.path.insert(0,f"{BASE}")
from features import build_features
from models_v2 import TCN
os.makedirs(f"{BASE}/training/checkpoints",exist_ok=True)

print(f"[{os.getpid()}] Loading data...",flush=True)
all_X,all_y=[],[]
for f in sorted(glob.glob(f"{BASE}/data/*/*/historical.csv")):
    try:
        parts=f.split("/");df=pd.read_csv(f,parse_dates=["timestamp"])
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
del X,y,perm

device="cuda"
m=TCN(Xt.shape[2],256,6,96).to(device)
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5)
crit=torch.nn.CrossEntropyLoss()
best,pat,t0=0,0,time.time()
B=2048

for ep in range(100):
    p=torch.randperm(len(Xt))
    for i in range(0,len(p),B):
        idx=p[i:i+B]
        xb=torch.FloatTensor(Xt[idx]).to(device,non_blocking=True)
        yb=torch.LongTensor(yt[idx]).to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True);crit(m(xb),yb).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    m.eval()
    with torch.no_grad():
        xv=torch.FloatTensor(Xv).to(device);yv_=torch.LongTensor(yv).to(device)
        vl=crit(m(xv),yv_).item();acc=(m(xv).argmax(1)==yv_).float().mean().item()
    m.train();sched.step(vl)
    if acc>best:best=acc;pat=0;torch.save({"state":m.state_dict(),"val_acc":acc},f"{BASE}/training/checkpoints/tcn_best.pth")
    else:pat+=1
    vram=torch.cuda.memory_allocated(0)/1024/1024/1024
    print(f"TCN EP{ep:>3} | VL:{vl:.4f} | Acc:{acc:.3f} | Best:{best:.3f} | VRAM:{vram:.2f}GB | {time.time()-t0:.0f}s",flush=True)
    if pat>=15:print(f"Early stop EP{ep}",flush=True);break
print(f"TCN Done: Best={best:.3f}",flush=True)