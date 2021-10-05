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

def make_clean_df(df:pd.DataFrame):
    d=dict(
        application_date=make_date_series(df['Application date']),
        countries=make_country_series(df['Applicant country of residence']),
        cpc=make_cpc_series(df['CPC (invention information)'],df['CPC (additional information)']),
        publication=df['Publication']
    )
    return pd.DataFrame(d)

def save_clean(path='../dane/clean.pickle'):
    with open(path,'bw') as f:
        pickle.dump(clean_df,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_clean(path='../dane/clean.pickle'):
    with open(path,'br') as f:
        return pickle.load(f)

def fractions_countries(df:pd.DataFrame):
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
    return pd.DataFrame(kody_krajow, columns=list(map(str,lista_krajow)))



if __name__ == '__main__':
    df = merge_files('../dane/*.csv')
    clean_df = make_clean_df(df)
    with open('../dane/clean.pickle','bw') as f:
        pickle.dump(clean_df,f,protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_clean()
    assert (loaded == clean_df).all().all()
