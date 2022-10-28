import numpy as np, pandas as pd, matplotlib.pyplot as plt

all_sm, all_xl = (pd.read_csv(f'data/gc_explain/all_{SIZE}.csv').sort_values(by='objective').reset_index(drop=True) for SIZE in ['SM', 'XL'])
sm_to_xl, xl_to_sm = [], []
cols = [f'p{i}' for i in range(6)]
idxes = [_ for _ in range(1000)]
for idx in idxes:
    xl_d = all_xl.iloc[idx]
    sm_d = all_sm.iloc[idx]
    look_xl = tuple(xl_d[cols])
    look_sm = tuple(sm_d[cols])
    sm_to_xl.append(np.where((all_xl[cols] == look_sm).sum(1) == 6)[0][0])
    xl_to_sm.append(np.where((all_sm[cols] == look_xl).sum(1) == 6)[0][0])

fig, ax = plt.subplots()
ycap = 1000
#ax.scatter(idxes, sm_to_xl, label='SM ranks --> XL ranks', marker='.')
ax.scatter([i for i,j in zip(idxes, sm_to_xl) if j > ycap], [d for d in sm_to_xl if d > ycap], label='SM ranks --> XL ranks', marker='.')
#ax.scatter(idxes, xl_to_sm, label='XL ranks --> SM ranks', marker='.')
ax.scatter([i for i,j in zip(idxes, xl_to_sm) if j > ycap], [d for d in xl_to_sm if d > ycap], label='XL ranks --> SM ranks', marker='.')
ax.set_ylabel('RIGHT RANK')
ax.set_xlabel('LEFT RANK')
ax.legend()
#ax.set_ylim([0,ycap])
ax.set_ylim([ycap,10648])
ax.set_xlim([0,max(idxes)+1])
ax.set_title(f"Omit {len([_ for _ in sm_to_xl if _ <= ycap])} SM->XL, {len([_ for _ in xl_to_sm if _ <= ycap])} XL->SM")
plt.savefig('sm_xl.png')

