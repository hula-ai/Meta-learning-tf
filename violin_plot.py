import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

methods = {"MAML": "./MAML_tensorflow/logs/omniglot_5way2shot",
           "MANN": "./NTM_tensorflow/saved_model"}
test_types = {"Regular": "",
              "Uniform_noise_0.3": "_uniform0.30",
              "Uniform_noise_0.5": "_uniform0.50",
              "Random_noise_0.3": "_random0.30",
              "Random_noise_0.5": "_random0.50",
              "Reduce_spt_0.3": "_reduce0.30",
              "Reduce_spt_0.5": "_reduce0.50"}

all_df = []
for mkey in methods:
    prefix = "cls_5.mbs_32.ubs_2.numstep1.updatelr0.4batchnorm" if mkey == "MAML" else "MANN_5way2shot"
    for tkey in test_types:
        result_folder = os.path.join(methods[mkey], prefix + test_types[tkey])
        result_file = os.path.join(result_folder, "test_results.npz")
        accs = np.load(result_file)["accs"]
        if mkey == "MAML":
            accs = accs[:, -1]
        data = {"accs": accs, "method": mkey, "test_type": tkey}
        df = pd.DataFrame(data)
        all_df.append(df)
all_data = pd.concat(all_df, ignore_index=True)

sns.set(style="whitegrid")
ax = sns.violinplot(x="test_type", y="accs", hue="method",
                    data=all_data, palette="muted", split=True)
ax.set_xticklabels(ax.get_xticklabels(),
                   rotation=65,
                   fontsize="xx-small",
                   horizontalalignment='right',)
plt.savefig("violin_plot.png", bbox_inches="tight", dpi=200)











