import data
import pandas as pd

if __name__ == '__main__':
    clean_df=data.load_clean()
    fractions = data.fractions_countries(clean_df, with_others=True)
    total = fractions.sum()

    total.to_csv('../gen/geo.csv')
    ...