import pickle
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file = "/cvhci/temp/prgc/hdgcn_filtered/work_dir_1/ntu_hdgcn/cross-view/bone_CoM_1/epoch50_test_score.pkl"
with open(file, "rb") as f:
    dict = pickle.load(f)

y = []
data = []
for k in dict.keys():
    print(k)

for k in dict.keys():
    
    action_class = int(k[k.find("A") + 1 : k.find("A") + 4])
    y.append(action_class)
    data.append(dict[k])

np_data = np.array(data, dtype=np.float32)

tsne = TSNE(n_components=2, verbose=1, random_state=123, n_iter=1500, metric="cosine")
z = tsne.fit_transform(np_data)

df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]

sns.scatterplot(
    x="comp-1",
    y="comp-2",
    hue=df.y.tolist(),
    palette=sns.color_palette("hls", 60),
    s=5,
    data=df,
    legend=False,
)

plt.savefig(file.split(".")[0] + ".png", dpi=900, bbox_inches='tight')