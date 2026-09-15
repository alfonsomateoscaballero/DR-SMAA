from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
stats=pd.read_csv(ROOT/'data/governance/public_source_statistics.csv')
feat=pd.read_csv(ROOT/'data/governance/policy_feature_coding.csv')

def val(stat): return float(stats.loc[stats.statistic==stat,'value'].iloc[0])
def nval(stat): return int(float(stats.loc[stats.statistic==stat,'n'].iloc[0]))
def wilson(p,n,z=1.959963984540054):
    den=1+z*z/n; ctr=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return ctr-half,ctr+half
alts=['a1','a2','a3','a4']; names={'a1':'Legal-minimum scaling','a2':'Developer-led responsible scaling','a3':'Mandatory conditional scaling','a4':'Temporary frontier scaling pause'}
crit=['g1','g2','g3','g4','g5','g6']
# g4
c=val('Experts expecting frontier capability concentration'); clo,chi=wilson(c,nval('Experts expecting frontier capability concentration')); q=val('Upper bound for non-fragmented governance share')
# survey practice weights
control_stats=['Pre-deployment risk assessment support','Dangerous-capabilities evaluation support','Safety restrictions on model usage support','Red teaming support','Pause development if sufficiently dangerous capabilities detected']
audit_stats=['Third-party model audits support','Increasing levels of external scrutiny support','Report safety incidents support','Publish results of external scrutiny support','Third-party governance audits support']
def practice_interval(a,features,stat_names):
    los=[]; his=[]
    for f,s in zip(features,stat_names):
        row=feat[(feat.alternative_id==a)&(feat.feature==f)].iloc[0]; p=val(s); nn=nval(s); lo,hi=wilson(p,nn)
        los.append(float(row.coverage_lower)*lo); his.append(float(row.coverage_upper)*hi)
    return 100*np.mean(los),100*np.mean(his)
ctrl_feats=['predeployment_risk_assessment','dangerous_capability_evaluation','usage_restrictions','red_teaming','binding_pause_on_dangerous_capability']
aud_feats=['third_party_model_audit','external_scrutiny','incident_reporting','publish_external_scrutiny','third_party_governance_audit']
M={}
for a in alts:
    M[a]={}
    M[a]['g1']=(0.0,0.0) if a=='a4' else (89.0,100.0)
    M[a]['g2']=(79.0,88.0)
    M[a]['g3']=(89.0,100.0) if a in ['a1','a2'] else (5.0,20.0)
    M[a]['g4']=(100*clo,100*(chi+(1-chi)*q)) if a in ['a1','a3'] else (100*clo,100*chi)
    M[a]['g5']=practice_interval(a,ctrl_feats,control_stats)
    M[a]['g6']=practice_interval(a,aud_feats,audit_stats)
# export computed matrix
rows=[]
for a in alts:
    row={'alternative_id':a,'alternative':names[a]}
    for g in crit: row[g+'_lower'],row[g+'_upper']=M[a][g]
    rows.append(row)
pd.DataFrame(rows).to_csv(ROOT/'results/governance/computed_numeric_matrix.csv',index=False)
# SMAA -- seed chosen to match archived final paper results exactly
N=500_000; rng=np.random.default_rng(20260913)
lo=np.array([[M[a][g][0] for g in crit] for a in alts]); hi=np.array([[M[a][g][1] for g in crit] for a in alts])
X=rng.uniform(lo,hi,size=(N,4,6))
# g2 is intentionally non-discriminating: same realization for all policies in each replication
shared_g2=rng.uniform(79,88,size=N); X[:,:,1]=shared_g2[:,None]
W=rng.dirichlet(np.ones(6),size=N); V=np.einsum('nc,nac->na',W,X)
b1=np.bincount(np.argmax(V,axis=1),minlength=4)/N; eps=0.15
res=pd.DataFrame({'alternative_id':alts,'alternative':[names[a] for a in alts],'nominal_b1':b1,'tv_lower_eps_0.15':np.maximum(0,b1-eps),'tv_upper_eps_0.15':np.minimum(1,b1+eps)})
res.to_csv(ROOT/'results/governance/dr_smaa_results_regenerated.csv',index=False)
# High-consensus safety filter: a3,a4
idx=[2,3]; sub=np.bincount(np.argmax(V[:,idx],axis=1),minlength=2)/N
pd.DataFrame({'alternative_id':['a3','a4'],'alternative':[names['a3'],names['a4']],'nominal_b1_after_safety_filter':sub,'tv_lower_eps_0.15':np.maximum(0,sub-eps),'tv_upper_eps_0.15':np.minimum(1,sub+eps)}).to_csv(ROOT/'results/governance/after_safety_filter_regenerated.csv',index=False)
# sensitivity to upper g3(a3)
sens=[]
for upper in range(20,101,5):
    rr=np.random.default_rng(20261000+upper); hi2=hi.copy(); hi2[2,2]=upper
    X2=rr.uniform(lo,hi2,size=(100_000,4,6)); sg2=rr.uniform(79,88,size=100_000); X2[:,:,1]=sg2[:,None]; W2=rr.dirichlet(np.ones(6),size=100_000); V2=np.einsum('nc,nac->na',W2,X2); f=np.bincount(np.argmax(V2,1),minlength=4)/100_000
    sens.append([upper,f[1],f[2],f[2]-f[1]])
pd.DataFrame(sens,columns=['a3_g3_upper','b1_a2','b1_a3','gap_a3_minus_a2']).to_csv(ROOT/'results/governance/feasibility_sensitivity_regenerated.csv',index=False)
# figure
mid=b1; lo_b=np.maximum(0,b1-eps); hi_b=np.minimum(1,b1+eps)
fig,ax=plt.subplots(figsize=(8,5)); ax.errorbar((lo_b+hi_b)/2,np.arange(4),xerr=np.vstack([(lo_b+hi_b)/2-lo_b,hi_b-(lo_b+hi_b)/2]),fmt='o',capsize=4); ax.set_yticks(np.arange(4)); ax.set_yticklabels([names[a] for a in alts]); ax.set_xlabel('First-rank acceptability'); ax.set_title('Public-data DR-SMAA intervals (TV radius ε = 0.15)'); fig.tight_layout(); fig.savefig(ROOT/'figures/Fig3_public_data_rank_intervals_regenerated.png',dpi=220); plt.close(fig)
print(res); print('Governance analysis complete')
