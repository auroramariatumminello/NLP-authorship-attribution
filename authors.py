#%%
from wikidata_ent import WikidataEntity
import pandas as pd
authors = pd.read_csv("data/authors_us.csv")

#%%
api = WikidataEntity('Edgar Allan Poe')