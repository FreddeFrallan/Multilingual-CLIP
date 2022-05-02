import os
import pandas as pd
import numpy as np

n = 150000  # Chunk size. How many obs to translate per language.

df_blip = pd.read_csv("data/fine_tune_languages.csv", index_col=None)
df = pd.read_feather("data/ccs_synthetic_sv.feather")
df = df[["caption", "caption_sv", "url", "index"]]

df2 = pd.DataFrame(np.repeat(df_blip.to_numpy(), n, axis=0), columns=df_blip.columns)
df = pd.concat([df, df2], axis=1)

df["caption_multi"] = None
df = df.rename(
    columns={"language_code": "multi_language_code", "language_name": "multi_language_name"}
)
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
df["multi_target"] = df["multi_target"].astype("Int64")

df.loc[df["multi_language_code"] == "en", "caption_multi"] = df.loc[
    df["multi_language_code"] == "en", "caption"
]


df_list = [df[i : i + n].reset_index(drop=True) for i in range(0, len(df), n)]

os.makedirs("data_multi", exist_ok=True)
for i in range(0, len(df_list)):
    code = df_list[i]["multi_language_code"][0]
    part_num = str(i).zfill(3)
    df_list[i].to_feather(f"data_multi/{part_num}_ccs_synthetic_{code}.feather")


df.to_feather("data/ccs_synthetic_multi.feather")
