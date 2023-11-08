import pandas as pd
import plotly.express as plt
#import chart_studio.tools as tls
from pareto import pareto_frontier_lists_x_y as pareto_frontier
from pareto import pareto_fronts
import sys

dfile = sys.argv[1]

df = pd.read_csv(dfile)

df = df.sort_values(['mat2vec'])
print(df.head())
print(df.shape)
print(len(df.leaf.values))
print(len(df.mat2vec.values))

# get first pareto
pareto, _ = pareto_frontier(df.leaf.values, df.mat2vec.values, maxX=False, maxY=False)
ifpareto = ['+' if i in pareto else '-' for i in df.leaf]
print(len(ifpareto))
df['if_pareto'] = ifpareto

# get pareto fronts
paretox, paretoy = pareto_fronts(df.leaf.values, df.mat2vec.values, maxX=False, maxY=False, nfronts=168)
n_pareto = []
for x,y in zip(df.leaf, df.mat2vec):
    for i in range(len(paretox)):
        if x in paretox[i] and y in paretoy[i]:
            n_pareto.append(i+1)

df['pareto_front'] = n_pareto[:-27] 

elements = 'Li,Be,Na,Mg,Al,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Rb,Sr,Y,Zr,Nb,Mo,Ag,Cd,In,Sn,Cs,Ba,Hf,Ta,W,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi'.split(',')
els = []

for i in df['phases']:
    ans = sorted(list(set(elements).intersection(set(i.split()))))
    els.append(ans[0])

df['elements'] = els

df = df[df['pareto_front']<=30]

df.to_csv(f'{dfile.split(".csv")[0]}_plotlychart.csv', index=False)

fig = plt.scatter(df, x='leaf',y='mat2vec', marginal_x='histogram',
        marginal_y='histogram', color='pareto_front', hover_name='phases', symbol='elements') #,'VAE','P2V','pareto_front']) #'if_pareto')
#fig.update_traces(marker_size=12)
fig.update_layout(coloraxis_colorbar_x=-0.15)
fig.update_layout(showlegend=True)
fig.update_layout(title="Synthetic accessibility for ternary intermetallics")
fig.show()
fig.write_html(f'{dfile.split(".csv")[0]}_plotly.html')

