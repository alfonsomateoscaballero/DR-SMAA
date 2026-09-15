from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'data/synthetic/synthetic_parameters.json').read_text())
rng=np.random.default_rng(P['seed'])
alternatives=P['alternatives']; m=len(alternatives); n=P['criteria_count']; N=P['N']; tau=P['tau']; alpha=P['alpha']
stress=rng.beta(P['stress_distribution']['a'],P['stress_distribution']['b'],size=N)
base=np.array(P['base_scores']); effect=np.array(P['stress_effect'])
X=np.clip(base[None,:,:]+stress[:,None,None]*effect[None,:,:]+rng.normal(0,P['criterion_noise_sd'],size=(N,m,n)),0,1)
mean_w=np.array(P['mean_weight']); W=rng.dirichlet(mean_w*P['dirichlet_concentration'],size=N)
V=np.einsum('kn,kmn->km',W,X)
order=np.argsort(-V,axis=1); ranks=np.empty_like(order); ranks[np.arange(N)[:,None],order]=np.arange(1,m+1)
first=(ranks==1)
risk_base=np.array(P['risk_base']); risk_slope=np.array(P['risk_slope']); risk_sd=np.array(P['risk_sd'])
Q=np.clip(risk_base[None,:]+stress[:,None]*risk_slope[None,:]+rng.normal(0,risk_sd,size=(N,m)),0,1)
safe=(Q<=tau)
nom_first=first.mean(0); nom_safe=safe.mean(0)
pd.DataFrame({'alternative':alternatives,'nominal_first_rank':nom_first,'nominal_safety':nom_safe}).to_csv(ROOT/'results/synthetic/nominal.csv',index=False)

def tv_event_bounds(indicator,eps):
    p=float(indicator.mean())
    if indicator.all(): return 1.0,1.0
    if (~indicator).all(): return 0.0,0.0
    return max(0.0,p-eps),min(1.0,p+eps)
rows=[]
for eps in np.linspace(0,0.30,61):
    for i,a in enumerate(alternatives):
        lo,hi=tv_event_bounds(first[:,i],eps); slo,_=tv_event_bounds(safe[:,i],eps)
        rows.append([eps,a,lo,hi,slo])
pd.DataFrame(rows,columns=['epsilon_tv','alternative','lower_b1','upper_b1','RSAI']).to_csv(ROOT/'results/synthetic/tv_results_corrected.csv',index=False)

# 50-cluster Wasserstein sensitivity approximation
H=P['cluster_aggregated_wasserstein']['H']
edges=np.quantile(stress,np.linspace(0,1,H+1)); cluster=np.clip(np.digitize(stress,edges[1:-1],right=True),0,H-1)
p0=np.bincount(cluster,minlength=H).astype(float)/N
stress_h=np.array([stress[cluster==h].mean() for h in range(H)])
first_h=np.vstack([first[cluster==h].mean(axis=0) for h in range(H)])
safe_h=np.vstack([safe[cluster==h].mean(axis=0) for h in range(H)])
sn=(stress_h-stress_h.min())/(stress_h.max()-stress_h.min()); C=np.abs(sn[:,None]-sn[None,:])
A_eq=np.zeros((H,H*H))
for h in range(H): A_eq[h,h*H:(h+1)*H]=1
cost=C.reshape(-1)
def w_bound(event_rates,eps,maximize=False):
    obj=np.tile(event_rates,H)
    res=linprog(-obj if maximize else obj,A_ub=cost.reshape(1,-1),b_ub=[eps],A_eq=A_eq,b_eq=p0,bounds=(0,None),method='highs')
    val=-res.fun if maximize else res.fun
    return max(0.0,min(1.0,float(val)))
out=[]
for eps in [0,0.01,0.025,0.05,0.10]:
    for i,a in enumerate(alternatives): out.append([eps,a,w_bound(first_h[:,i],eps),w_bound(first_h[:,i],eps,True),w_bound(safe_h[:,i],eps)])
pd.DataFrame(out,columns=['epsilon_w','alternative','lower_b1','upper_b1','RSAI']).to_csv(ROOT/'results/synthetic/wasserstein_cluster_aggregated.csv',index=False)

# CAR/SAR diagnostics for fixed-support TV
winner=int(np.argmax(nom_first)); car_rows=[]
for j in range(m):
    if j==winner: continue
    if first[:,j].any(): car=(nom_first[winner]-nom_first[j])/2
    else: car=nom_first[winner]
    car_rows.append([alternatives[winner],alternatives[j],car])
pd.DataFrame(car_rows,columns=['winner','competitor','CAR_TV']).to_csv(ROOT/'results/synthetic/car_pairwise.csv',index=False)
sar_rows=[]
for i,a in enumerate(alternatives):
    if nom_safe[i] < alpha: status='already_inadmissible_at_epsilon_0'; sar=np.nan
    elif safe[:,i].all(): status='not_reached_under_fixed_support_reweighting'; sar=np.nan
    else: status='finite'; sar=nom_safe[i]-alpha
    sar_rows.append([a,nom_safe[i],status,sar])
pd.DataFrame(sar_rows,columns=['alternative','nominal_safety','SAR_status','SAR_TV']).to_csv(ROOT/'results/synthetic/sar.csv',index=False)
print('Synthetic benchmark complete')
