from functools import cached_property

import networkx as nx
import numpy as np
from matplotlib.pyplot import cla
import pandas as pd
import glob
import tuples
import pickle

def merge_files(pattern = '/Users/agnieszka/OneDrive/PhD/KR_dzielone/AI_patenty/dane/*.csv' ):
    tables=[
        pd.read_csv(f,sep=';') for f in glob.glob(pattern)
    ]
    return pd.concat(tables).drop_duplicates(subset="Publication").reset_index()

def iter_lines(ds:pd.Series, by_line=True):
    for applicants in ds:
        if hasattr(applicants,'split'):
            la =  applicants.split('\n')
        else:
            la=[]
        if by_line:
            for applicant in la:
                yield applicant
        else:
            yield la

_Unia=['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE' ,'GR' ,'HU', 'IE','IT', 'LV', 'LT', 'LU', 'MT' ,'NL', 'PL' ,'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB']



Unia = [tuples.Country(c) for c in _Unia]

def _make_list(o):
    if not isinstance(o,list):
        return [o]
    return o

def make_country_series(ds:pd.Series):
    countries = []
    for cs in iter_lines(ds, by_line=False):
        cs = _make_list(cs)
        new_cs = [tuples.Country(c) if c in _Unia else tuples.Country('Other') for c in cs]
        countries.append(new_cs)
    return pd.Series(countries, index=ds.index)



def make_cpc_series(invention:pd.Series, additional:pd.Series):
    cpcs = []
    for cpci, cpca in zip(
        iter_lines(invention, by_line=False),
        iter_lines(additional, by_line=False)):


        cpci = _make_list(cpci)
        cpca = _make_list(cpca)

        lcpc = list(map(tuples.CPC.parse,cpci)) + list(map(tuples.CPC.parse,cpca))
        cpcs.append(lcpc)

    return pd.Series(cpcs, index=invention.index)

def make_date_series(ds:pd.Series):
    return  pd.to_datetime(ds,format='%Y%m%d')

def make_clean_df(df:pd.DataFrame, country_column='Applicant country of residence'):
    d=dict(
        application_date=make_date_series(df['Application date']),
        countries=make_country_series(df[country_column]),
        cpc=make_cpc_series(df['CPC (invention information)'],df['CPC (additional information)']),
        publication=df['Publication'],
        publication_date = make_date_series(df['Publication date'])
    )
    return pd.DataFrame(d)

def save_clean(path='../dane/clean.pickle'):
    with open(path,'bw') as f:
        pickle.dump(clean_df,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_clean(path='../dane/clean.pickle'):
    with open(path,'br') as f:
        return pickle.load(f)

def fractions_countries(df:pd.DataFrame, with_others=True):
    others = tuples.Country('Others')
    lista_krajow=Unia + [others]
    kody_krajow=[]
    for kraje in df['countries']:
        z = np.zeros(len(lista_krajow))
        for kraj in kraje:
            kraj = others if not kraj in Unia else kraj
            z[lista_krajow.index(kraj)] += 1.
            z /= z.sum()
        kody_krajow.append(z)
    if with_others:
        return pd.DataFrame(kody_krajow, columns=list(map(str,lista_krajow)))
    return pd.DataFrame(kody_krajow, columns=list(map(str, lista_krajow))).iloc[:,:-1]


def graph(df:pd.DataFrame, cpc_level=1)->nx.Graph:
    edges=[]
    for row in df.itertuples():
        p = tuples.Patent(id=row.publication,
                          t=row.application_date,
                          i=row.Index)
        for c in row.countries:
            edges.append((p, c))
        for cpc in row.cpc:
            edges.append((p, cpc.at_level(cpc_level)))

    G = nx.Graph(edges)
    return G

class AICPC:
    @cached_property
    def ai_cpc(self):
        dodatkowecpc = '''G06N5/003
        G06N5/006
        G06N5/02
        G06N5/022
        G06N5/025
        G06N5/027
        G06N5/04
        G06N5/00
        G06N5/041
        G06N5/042
        G06N5/043
        G06N5/045
        G06N5/046
        G06N5/047
        G06N5/048
        G06N7/005
        G06N7/02
        G06N7/00
        G06N7/023
        G06N7/026
        G06N7/04
        G06N7/043
        G06N7/046
        G06N7/06
        G06N7/08
        G06N20/10
        G06N20/20
        G06N20/00'''.split('\n')
        CPC_AI = ['A61B5/7267', 'G01N33/0034', 'G06F19/24', 'G10H2250/151', 'H04L2025/03464', 'B29C66/965',
                  'G01N2201/1296', 'G06F19/707', 'G10H2250/311', 'H04N21/4662', 'B29C2945/76979', 'G01S7/417',
                  'G06F2207/4824', 'G10K2210/3024', 'H04N21/4663', 'B60G2600/1876', 'G05B13/027', 'G06K7/1482',
                  'G10K2210/3038', 'H04N21/4665', 'B60G2600/1878', 'G05B13/0275', 'G06N3/004', 'G10L25/30',
                  'H04N21/4666', 'B60G2600/1879', 'G05B13/028', 'G06N3/02', 'G11B20/10518', 'H04Q2213/054',
                  'E21B2041/0028', 'G05B13/0285', 'G06N3/12', 'H01J2237/30427', 'H04Q2213/13343', 'F02D41/1405',
                  'G05B13/029', 'H02P21/0014', 'H04Q2213/343', 'F03D7/046', 'G05B13/0295',
                  'H02P23/0018', 'H04R25/507', 'F05B2270/707', 'G05B2219/33002', 'H03H2017/0208',
                  'F05B2270/709', 'G05D1/0088', 'G06N99/005', 'H03H2222/04',
                  'F05D2270/707', 'G06F11/1476', 'G06T3/4046', 'H04L25/0254', 'F05D2270/709',
                  'G06F11/2257', 'G06T9/002', 'H04L25/03165', 'F16H2061/0081', 'G06F11/2263', 'G06T2207/20081',
                  'H04L41/16', 'F16H2061/0084', 'G06F15/18', 'G06T2207/20084', 'H04L45/08', 'G01N29/4481',
                  'G06F17/16', 'G08B29/186', 'H04L2012/5686','Y10S128/924', 'Y10S128/925'] + [d.strip() for d in dodatkowecpc]
        CPC_AI_ = [cpc[:4] + ' ' + cpc[4:] for cpc in CPC_AI]
        CPC_AI_O = [tuples.CPC.parse(cpc) for cpc in CPC_AI_]

        return  CPC_AI_O

    def counts(self,df:pd.DataFrame)->dict:
        ai_cpcs = set(self.ai_cpc)
        counts = {k:0 for k in ai_cpcs}
        for paten_cpcs in df.cpc:
            for cpc in set(paten_cpcs).intersection(ai_cpcs):
                counts[cpc]+=1
        return counts

if __name__ == '__main__':
    df = merge_files('../dane/*.csv')
    clean_df = make_clean_df(df)
    with open('../dane/clean.pickle','bw') as f:
        pickle.dump(clean_df,f,protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_clean()
    assert (loaded == clean_df).all().all()
