import pandas as pd
import json

with open("data/ccs_synthetic_filtered_large.json") as f:
    d = json.load(f)

df = pd.DataFrame(d)
df["index"] = df.index + 1
df["nr_words"] = df["caption"].apply(lambda x: len(x.split()))

df.to_feather("data/ccs_synthetic.feather")
