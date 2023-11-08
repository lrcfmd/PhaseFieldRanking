import pandas as pd
import plotly.express as plt
import sys
from calculator_pareto import calculate_pareto_fronts 

dfile = sys.argv[1]

df = pd.read_csv(dfile)

# Dictionary pair of values: phase
dic = {' '.join([str(l), str(m)]): phase for l, m, phase in zip(df.leaf, df.mat2vec, df.phases)}  
print('Dictionary created')

# get pareto fronts
print('Calculating pareto fronts')
pareto_fronts = calculate_pareto_fronts(df.leaf.values, df.mat2vec.values)
print('Pareto calculated')

fronts = []
for n, front in enumerate(pareto_fronts):
       print(f"Pareto Front {n + 1}:")
       front_points = [[df.leaf.values[j], df.mat2vec.values[j]] for j in front]
       keys = [' '.join(list(map(str, point))) for point in front_points]
       phases = [dic[key] for key in keys]
       bf = pd.DataFrame({'phases': phases, 'leaf': [point[0] for point in front_points], 
           'mat2vec': [point[1] for point in front_points], 'pareto_front': [n+1 for i in range(len(front_points))]})
       fronts.append(bf)

cf = pd.concat(fronts)


elements = 'Li,Be,Na,Mg,Al,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Rb,Sr,Y,Zr,Nb,Mo,Ag,Cd,In,Sn,Cs,Ba,Hf,Ta,W,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi'.split(',')
els = []

for i in cf['phases']:
    ans = sorted(list(set(elements).intersection(set(i.split()))))
    els.append(ans[0])

cf['elements'] = els
cf.to_csv(f'{dfile.split(".csv")[0]}_plotlychart.csv', index=False)

fig = plt.scatter(cf, x='leaf',y='mat2vec', marginal_x='histogram',
        marginal_y='histogram', color='pareto_front', hover_name='phases', symbol='elements') #,'VAE','P2V','pareto_front']) #'if_pareto')
#fig.update_traces(marker_size=12)
fig.update_layout(coloraxis_colorbar_x=-0.15)
fig.update_layout(showlegend=True)
fig.update_layout(title="Synthetic accessibility for ternary intermetallics")
fig.show()
fig.write_html(f'{dfile.split(".csv")[0]}_plotly.html')

