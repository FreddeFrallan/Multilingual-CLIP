import os
import pandas as pd

filenames = os.listdir("data_translated")
df = pd.read_feather("data/ccs_synthetic_multi.feather")
df_list = [pd.read_feather(os.path.join("data_translated", filename)) for filename in filenames]
df_multi = pd.concat(df_list)
df_multi = df_multi.reset_index(drop=True)

df = df.drop("caption_multi", axis=1)
df = df.merge(df_multi[["caption_multi", "index"]], how="left", on="index")

df = df[
    [
        "caption",
        "caption_sv",
        "caption_multi",
        "url",
        "multi_language_code",
        "multi_language_name",
        "multi_target",
        "target_code",
        "opus_mt_url",
        "index",
    ]
]

df = df.rename(columns={"multi_target": "multiple_target_model"})
df["opus_mt_url"] = df["opus_mt_url"].str.replace("https://huggingface.co/", "")

df.to_feather("ccs_synthetic.feather")
