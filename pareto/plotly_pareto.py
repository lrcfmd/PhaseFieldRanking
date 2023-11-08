import pandas as pd
import plotly.express as plt
#import chart_studio.tools as tls
from pareto import pareto_frontier_lists_x_y as pareto_frontier
from pareto import pareto_fronts
import sys

dfile = sys.argv[1]

df = pd.read_csv(dfile)

print(df.head())

anions = ['O', 'S', 'Br', 'I', 'Cl', 'F', 'N']

ans = []
for i in df['phases']:
    an = list(set(anions).intersection(set(i.split())))
    ans.append(an[0])

df['anion'] = ans

fig = plt.scatter(df, x='leaf',y='mat2vec', marginal_x='histogram',
        marginal_y='histogram',color='pareto_front', symbol='anion', hover_name='phases')
#fig.update_traces(marker_size=12)
fig.update_layout(coloraxis_colorbar_x=-0.15)
fig.update_layout(showlegend=True)
fig.update_layout(title="Synthetic accessibility for Mg-M-M'-A")
fig.show()
fig.write_html(f'{dfile.split(".csv")[0]}.html')
