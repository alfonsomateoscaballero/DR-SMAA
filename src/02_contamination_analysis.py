from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
rep=pd.read_csv(ROOT/'data/contamination/random_contamination_replications.csv')
# Historical column false_robustness is semantically oracle disagreement; retain raw file unchanged, rename in analysis.
rep['oracle_disagreement'] = rep['false_robustness']
rep['correct_and_safe'] = ((rep['accuracy']==1) & (rep['safety_violation']==0)).astype(int)

def wilson(k,n,z=1.959963984540054):
    p=k/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,ctr-half), min(1,ctr+half)
metrics=[]
for (delta,method),d in rep.groupby(['delta','method']):
    for name,col in [('recommendation_accuracy','accuracy'),('oracle_disagreement_rate','oracle_disagreement'),('safety_violation_rate','safety_violation'),('correct_and_safe_rate','correct_and_safe'),('indecision_rate','indecision')]:
        k=int(d[col].sum()); n=len(d); lo,hi=wilson(k,n); metrics.append([delta,method,name,k/n,lo,hi,n])
out=pd.DataFrame(metrics,columns=['delta','method','metric','estimate','ci_low','ci_high','n'])
out.to_csv(ROOT/'results/contamination/metrics_with_ci.csv',index=False)
# figures
piv=out[out.metric=='safety_violation_rate']
fig,ax=plt.subplots(figsize=(8,5))
for method,d in piv.groupby('method'):
    ax.errorbar(d.delta,d.estimate,yerr=np.vstack([np.maximum(0,d.estimate-d.ci_low),np.maximum(0,d.ci_high-d.estimate)]),marker='o',capsize=3,label=method)
ax.set_xlabel('Contamination fraction δ'); ax.set_ylabel('Safety violation rate'); ax.set_title('Safety violations under distributional misspecification (95% Wilson CI)'); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/Fig1_safety_violation_rate_regenerated.png',dpi=220); plt.close(fig)
acc=out[out.metric=='recommendation_accuracy'][['delta','method','estimate']].rename(columns={'estimate':'accuracy'})
sv=out[out.metric=='safety_violation_rate'][['delta','method','estimate']].rename(columns={'estimate':'safety'})
m=acc.merge(sv,on=['delta','method'])
fig,ax=plt.subplots(figsize=(8,5))
for method,d in m.groupby('method'): ax.plot(d.safety,d.accuracy,marker='o',label=method)
ax.set_xlabel('Safety violation rate'); ax.set_ylabel('Recommendation accuracy'); ax.set_title('Accuracy-safety trade-off under contamination'); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/Fig2_accuracy_safety_tradeoff_regenerated.png',dpi=220); plt.close(fig)
print('Contamination analysis complete; frozen replications are the primary input.')
