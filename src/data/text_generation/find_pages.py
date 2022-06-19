import pandas as pd
from sqlalchemy import create_engine

from src.features import data_cleaning

page_df = pd.DataFrame({}, columns=["text", "isMerger", "isAcquisition", "isMergerOrAcquisition", "website"], )

i = 0
with open('assorted_pages.txt', 'r') as file:
    for line in file:
        if i <= 102:
            i += 1
            continue
        if i > 152:
            break
        url = str(line.split(' ')[-1][:-1])
        print(url)
        name = ' '.join(line.split(' ')[:-1])
        print(name)
        try:
            df = the_dry_cleaner.dry_clean_the_data([url], False, name)
            df['isMerger'][0] = 0
            df['isAcquisition'][0] = 0
            df['isMergerOrAcquisition'][0] = 0

            page_df = pd.concat([page_df, df])
            i += 1
        except Exception:
            print(f'Error parsing {name}')

engine = create_engine("sqlite:///../../../data/processed/test_train_database.db", echo=True)
connection = engine.connect()

page_df.to_sql('DryCleaned_EntirePage', connection, if_exists="append", index=False)
