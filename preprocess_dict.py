import pandas as pd

dictionary = pd.read_excel("dictionary.xlsx")

chars_zhuyin = dictionary.loc[(dictionary["字詞名"].str.len() == 1) & (~dictionary["注音一式"].isnull()), ["字詞名", "注音一式"]]
chars_zhuyin = chars_zhuyin.rename(columns={"字詞名": "char", "注音一式": "zhuyin"})

chars_zhuyin.to_csv("chars_zhuyin.csv", index=False)
