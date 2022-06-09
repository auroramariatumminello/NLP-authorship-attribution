# %%
import requests
from wikidata.client import Client
import pandas as pd    
import wikidata

class WikidataEntity :

    def __init__(self, title):
        self.client = Client()
        self.response = None
        self.title = title
        self.response = self.get_api_result(title)
        self.id = self.get_id_from_title()
        self.entity = self.get_wikidata_entity()
        self.aliases = self.get_aliases()
        self.description = self.get_description()
        

    def get_id_from_title(self):
        response = self.get_api_result(self.title)
        if response['entities']:
            return list(response['entities'].keys())[0]
        else:
            # There is no entity about it
            return None

    def get_api_result(self, title, is_title=True):
        if is_title:
            params = dict(
                        action='wbgetentities',
                        format='json',
                        languages='en',
                        sites='enwiki',
                        titles=title,
                        normalize=1
                    )
        else:
            params = dict(
                    action='wbgetentities',
                    format='json',
                    languages='en',
                    sites='enwiki',
                    ids=title,
                    normalize=1
                )
        while self.response is None:
            try:
                self.response = requests.get(
                    'https://www.wikidata.org/w/api.php?', params).json()
            except:
                print("Connection error, retrying...")
        return self.response

    def get_aliases(self):
        if 'aliases' in self.response['entities'][self.id] and 'en' in self.response['entities'][self.id]['aliases']:
            aliases = self.response['entities'][self.id]['aliases']['en']
            self.aliases = [x['value'] for x in aliases]
        else:
            self.aliases = None
        return self.aliases

    def get_description(self):
        if 'descriptions' in self.response['entities'][self.id] and 'en' in self.response['entities'][self.id]['descriptions']:
            self.description = self.response['entities'][self.id]['descriptions']['en']['value']
        else:
            self.description = None
        return self.description
    
    def get_wikidata_entity(self):
        self.entity = self.client.get(self.id)
        return self.entity if self.entity.id != "-1" else None
    
    def get_entity_properties(self):
        self.entity = self.get_wikidata_entity()
        if self.entity.id == "-1": # Entity not found
            return None
        try:
            self.properties = iter(self.entity.lists())
        except:
            self.properties = self.entity.iterlists()
        properties_df = []
        # How to get the property name
        while True:
            try:
                prop = next(self.properties)
                # Do not consider disambiguation links
                if prop[0].label['en'] == "different from" or " ID" in prop[0].label['en']:
                    continue
                else:
                    for value in prop[1]:
                        
                        # FILE
                        if isinstance(value, wikidata.commonsmedia.File):
                            # Do not consider images
                            continue
                        
                        # QUANTITY
                        elif isinstance(value, wikidata.quantity.Quantity):
                            # if the unit of measurement is not specified
                            if value.unit is None:
                                properties_df.append([prop[0].label['en'], prop[0].id, value.amount, None])
                            elif isinstance(value.unit, wikidata.entity.Entity):
                                properties_df.append([prop[0].label['en'], prop[0].id, str(value.amount)+" "+str(value.unit.label['en']), None])
                            else:
                                properties_df.append([prop[0].label['en'], prop[0].id, str(value.amount)+" "+value.unit, None])
                        
                        # MULTILINGUAL
                        elif isinstance(value, wikidata.multilingual.MultilingualText) or isinstance(value, wikidata.multilingual.MonolingualText):
                            continue
            
                        # ENTITY
                        elif isinstance(value, wikidata.entity.Entity):
                            # If the value is an entity, grab its id
                            properties_df.append([prop[0].label['en'], prop[0].id, value.label['en'], value.id])
                            
                        # GEOGRAPHICAL POSITION
                        elif isinstance(value, wikidata.globecoordinate.GlobeCoordinate):
                            continue
                        # OTHERWISE IT IS JUST DATA   
                        else:
                            properties_df.append([prop[0].label['en'], prop[0].id, value, None])
            
            except wikidata.datavalue.DatavalueError:
                # If there is a decoder error, skip it
                continue
            except KeyError: # due to the missing version in English
                # print(self.entity.label)
                continue
            except StopIteration:
                # if StopIteration is raised, break from loop
                break
        properties_df = pd.DataFrame(properties_df, columns = ['property', 'prop_id', 'value','val_id'])
        properties_df['entity'] = [self.entity.label['en']]*len(properties_df)
        return properties_df

# %%
