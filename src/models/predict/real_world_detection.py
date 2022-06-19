import asyncio
import os.path
import re
import sqlite3
from datetime import datetime
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, LongformerForSequenceClassification, \
    LongformerTokenizer, pipeline

from src.features import data_cleaning, link_grabber
from src.models.predict.calculate_confidence_score import calculate_sentence_confidence_score, \
    calculate_abstracted_confidence_score

database_name = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'live',
                             'liveMergersAcquisitionsDB.db')

MAIN_RAW_DATA_PATH = '../../../data/raw/Potential_M&A_Activities.csv'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


async def generate_results_db(url_list: list, company_name: str, sentence_table_name: str, webpage_rank_table_name: str,
                              company_rank_table_name: str, processed_webpages_table_name: str, connection,
                              page_tokenizer, page_model, sentence_tokenizer, general_sentence_model,
                              merger_sentence_model, acquisition_sentence_model, device: str, ner):
    """
    Scrapes data off of the internet given a set of urls, make predictions, and extract insights asynchronously

    :param url_list: list of urls
    :param company_name: name of queried company
    :param sentence_table_name: name of sentence table
    :param webpage_rank_table_name: name of webpage rank table
    :param company_rank_table_name: name of company rank table
    :param processed_webpages_table_name: name of processed webpages table
    :param connection: database connection
    :param page_tokenizer:page level tokenizer
    :param page_model: page level model
    :param sentence_tokenizer: sentence level tokenizer
    :param general_sentence_model: isMergerOrAcquisition sentence level model
    :param merger_sentence_model: isMerger sentence level model
    :param acquisition_sentence_model: isAcquisition sentence lvel model
    :param device: device to run models on: 'cpu' or 'conda:0'
    :param ner: named entity recognition model
    :return: None
    """
    global sanitized_data_df

    query = connection.execute(
        f"SELECT website From {processed_webpages_table_name} WHERE queriedCompany  = '{company_name}'")
    cols = [column[0] for column in query.description]
    processed_url_list = list(pd.DataFrame.from_records(data=query.fetchall(), columns=cols)['website'])

    # remove all processed urls from the list of urls to process
    url_list = [x for x in url_list if x not in processed_url_list]

    # Add all new pages to pages which have been processed
    processed_webpages = pd.DataFrame({})
    processed_webpages['website'] = url_list
    processed_webpages['queriedCompany'] = company_name
    processed_webpages['date_processed'] = datetime.today().strftime('%Y-%m-%d')

    processed_webpages.to_sql(processed_webpages_table_name, connection, if_exists="append", index=False)

    page_sanitized = the_dry_cleaner.dry_clean_the_data(url_list, False, company_name)
    page_text = list(page_sanitized["text"])

    if page_text:

        page_tokenized = page_tokenizer(page_text, padding=True, truncation=True, max_length=3072)

        # Create torch dataset
        page_dataset = Dataset(page_tokenized)
        page_model = page_model.to(device)
        page_trainer = Trainer(page_model)

        # Make prediction
        page_loader = DataLoader(page_dataset, batch_size=1, shuffle=False)
        raw_pred, _, _ = page_trainer.prediction_loop(page_loader, description="prediction")

        y_pred = np.argmax(raw_pred, axis=1)

        # Drop that column
        page_sanitized.drop('isMergerOrAcquisition', axis=1, inplace=True)
        page_sanitized['isMergerOrAcquisition'] = y_pred

        page_sanitized = page_sanitized.loc[page_sanitized['isMergerOrAcquisition'] == 1]

        sentences_sanitized = pd.DataFrame({})

        if len(page_sanitized) != 0:

            for index, row in page_sanitized.iterrows():
                raw_text_list = re.split("\. |(?:\n+ ?)+", row.get('text').replace("\t", "").replace("\r", ""), )
                stripped_text_list = [st.strip() for st in raw_text_list]
                df = pd.DataFrame(stripped_text_list, columns=["text"])
                df['website'] = row.get('website')
                df['company'] = row.get('company')
                sentences_sanitized = pd.concat([sentences_sanitized, df])

            sentence_text = list(sentences_sanitized["text"])

            sentence_tokenized = sentence_tokenizer(sentence_text, padding=True, truncation=True, max_length=512)

            # Create torch dataset
            sentence_dataset = Dataset(sentence_tokenized)
            general_sentence_model = general_sentence_model.to(device)
            sentence_trainer = Trainer(general_sentence_model)

            # Make prediction
            sentence_loader = DataLoader(sentence_dataset, batch_size=4, shuffle=False)
            raw_pred, _, _ = sentence_trainer.prediction_loop(sentence_loader, description="prediction")

            # Preprocess raw predictions
            y_pred = np.argmax(raw_pred, axis=1)

            sentences_sanitized['isMergerOrAcquisition'] = y_pred
            sentences_sanitized = sentences_sanitized.loc[sentences_sanitized['isMergerOrAcquisition'] == 1]
            sentence_text = list(sentences_sanitized["text"])
            sentences_sanitized['involvedCompany'] = None

            if sentence_text:

                entity_rec = ner(sentence_text)

                for x in range(len(sentence_text)):
                    entity_df = pd.DataFrame(entity_rec[x])
                    if not entity_df.equals(pd.DataFrame({})):
                        entity_df = entity_df.loc[entity_df['entity_group'] == 'ORG']
                        entity_df = entity_df.loc[entity_df['score'] > 0.95]
                        sentences_sanitized.loc[
                            sentences_sanitized['text'] == sentence_text[x], 'involvedCompany'] = pd.Series(
                            [list(entity_df.word.unique())] * len(sentences_sanitized))

                sentence_tokenized = sentence_tokenizer(sentence_text, padding=True, truncation=True, max_length=512)

                # Create torch dataset
                sentence_dataset = Dataset(sentence_tokenized)
                merger_sentence_model = merger_sentence_model.to(device)
                merger_sentence_trainer = Trainer(merger_sentence_model)

                # Make prediction
                merger_sentence_loader = DataLoader(sentence_dataset, batch_size=4, shuffle=False)
                raw_pred, _, _ = merger_sentence_trainer.prediction_loop(merger_sentence_loader,
                                                                         description="prediction")

                # Preprocess raw predictions
                y_pred = np.argmax(raw_pred, axis=1)

                sentences_sanitized['isMerger'] = y_pred

                acquisition_sentence_model = acquisition_sentence_model.to(device)
                acquisition_sentence_trainer = Trainer(acquisition_sentence_model)

                # Make prediction
                acquisition_sentence_loader = DataLoader(sentence_dataset, batch_size=4, shuffle=False)
                raw_pred, _, _ = acquisition_sentence_trainer.prediction_loop(acquisition_sentence_loader,
                                                                              description="prediction")

                # Preprocess raw predictions
                y_pred = np.argmax(raw_pred, axis=1)

                sentences_sanitized['isAcquisition'] = y_pred

                sentences_sanitized = sentences_sanitized.rename(columns={"company": "queriedCompany"})

                sentences_sanitized.loc[sentences_sanitized['involvedCompany'].isnull(), ['involvedCompany']] = \
                    sentences_sanitized.loc[sentences_sanitized['involvedCompany'].isnull(), 'involvedCompany'].apply(
                        lambda z: ["{}"])

                sentences_sanitized['confidenceScore'] = sentences_sanitized.apply(
                    lambda z: calculate_sentence_confidence_score(z['isMergerOrAcquisition'], z['isMerger'],
                                                                  z['isAcquisition'], z['involvedCompany']), axis=1)

                sentences_sanitized = pd.DataFrame(
                    {col: np.repeat(sentences_sanitized[col].values, sentences_sanitized['involvedCompany'].str.len())
                     for col in sentences_sanitized.columns.difference(['involvedCompany'])}).assign(
                    **{'involvedCompany': np.concatenate(sentences_sanitized['involvedCompany'].values)})[
                    sentences_sanitized.columns.tolist()]

                sentences_sanitized['involvedCompany'] = sentences_sanitized['involvedCompany'].replace('{}', None)

                sentences_sanitized.to_sql(sentence_table_name, connection, if_exists="append", index=False)

                # Generate webpage confidence rankings
                webpage_rankings = pd.DataFrame({})
                for company in sentences_sanitized['queriedCompany'].unique():
                    subset_sentences = sentences_sanitized.loc[
                        sentences_sanitized['queriedCompany'] == company
                        ]
                    for website in subset_sentences['website'].unique():
                        subset_sentences = sentences_sanitized.loc[
                            sentences_sanitized['website'] == website
                            ]

                        d = {
                            "confidenceScore": calculate_abstracted_confidence_score(subset_sentences),
                            "website": website,
                            "company": company
                        }

                        webpage_rankings = pd.concat(
                            (
                                webpage_rankings,
                                pd.DataFrame(
                                    [d],
                                    columns=d.keys()
                                )
                            ),
                            ignore_index=True,
                        )

                webpage_rankings.to_sql(webpage_rank_table_name, connection, if_exists="append", index=False)

                for company in webpage_rankings['company'].unique():
                    query = connection.execute(
                        f"SELECT * From {webpage_rank_table_name} WHERE company= '{company}'")
                    cols = [column[0] for column in query.description]
                    dataset = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
                    dataset['confidenceScore'] = pd.to_numeric(dataset['confidenceScore'])
                    score = calculate_abstracted_confidence_score(dataset)
                    connection.execute(f'INSERT INTO {company_rank_table_name} (company, confidenceScore) '
                                       f'VALUES(\'{company}\', {score}) '
                                       f'ON CONFLICT(company) DO UPDATE SET confidenceScore={score} ')


def apply_data(filename=None, names=None):
    """
    Generate Database Tables and Run Asynchronous Predictions

    :param filename: csv file name
    :param names: list of company names
    :return: None
    """
    conn = sqlite3.connect(database_name)

    sentence_table_name = 'Processed_Sentences'
    webpage_rank_table_name = 'Webpage_Confidence_Rankings'
    company_rank_table_name = 'Company_Confidence_Rankings'
    processed_websites_table_name = 'Processed_Websites'

    conn.execute(f'create table IF NOT EXISTS {sentence_table_name}'
                 f'('
                 f'text                  VARCHAR,'
                 f'isMergerOrAcquisition VARCHAR,'
                 f'isMerger              VARCHAR,'
                 f'isAcquisition         VARCHAR,'
                 f'website               VARCHAR,'
                 f'queriedCompany        VARCHAR,'
                 f'involvedCompany       VARCHAR,'
                 f'confidenceScore       VARCHAR'
                 f');')

    conn.execute(f'create table IF NOT EXISTS {webpage_rank_table_name}'
                 f'('
                 f'company               VARCHAR,'
                 f'website               VARCHAR,'
                 f'confidenceScore       VARCHAR'
                 f');')

    conn.execute(f'create table IF NOT EXISTS {company_rank_table_name}'
                 f'('
                 f'company               VARCHAR PRIMARY KEY,'
                 f'confidenceScore       VARCHAR'
                 f');')

    conn.execute(f'create table IF NOT EXISTS {processed_websites_table_name}'
                 f'('
                 f'queriedCompany              VARCHAR,'
                 f'website              VARCHAR,'
                 f'date_processed        VARCHAR'
                 f');')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Script is using device: " + device)

    sentence_model_name = "bert-base-uncased"
    sentence_tokenizer = BertTokenizer.from_pretrained(sentence_model_name)
    page_model_name = "allenai/longformer-base-4096"
    page_tokenizer = LongformerTokenizer.from_pretrained(page_model_name)
    # Load trained models
    page_model_path = model_dir('page_level_model_weights')
    page_model = LongformerForSequenceClassification.from_pretrained(page_model_path, num_labels=2)
    general_sentence_model_path = model_dir('merger_or_acquisition_sentence_level_model_weights')
    general_sentence_model = BertForSequenceClassification.from_pretrained(general_sentence_model_path, num_labels=2)
    merger_sentence_model_path = model_dir('merger_sentence_level_model_weights')
    merger_sentence_model = BertForSequenceClassification.from_pretrained(merger_sentence_model_path, num_labels=2)
    acquisition_sentence_model_path = model_dir('acquisition_sentence_level_model_weights')
    acquisition_sentence_model = BertForSequenceClassification.from_pretrained(acquisition_sentence_model_path,
                                                                               num_labels=2)
    ner = pipeline("ner", grouped_entities=True)

    search_keywords = ["{} acquisition", "{} merger"]
    if filename is not None:
        data_iter = link_grabber.grab_csv_links(filename, search_keywords)
    else:
        df = pd.DataFrame({'Mfr': names})
        data_iter = link_grabber.grab_links(df, search_keywords)

    for link_data in data_iter:
        # TODO: Remove previously queried links from link data (i.e. do not query the same link again and again)
        asyncio.run(
            generate_results_db(link_data["links"], link_data["company"], sentence_table_name, webpage_rank_table_name,
                                company_rank_table_name, processed_websites_table_name, conn, page_tokenizer,
                                page_model, sentence_tokenizer, general_sentence_model, merger_sentence_model,
                                acquisition_sentence_model, device, ner))


def model_dir(name):
    """
    Returns a path to a models directory based on its name

    :param name: name of models path
    :return: a path to a models directory
    """
    return os.path.relpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', name),
                           os.getcwd()).replace('\\', '/')


if __name__ == '__main__':
    freeze_support()
    apply_data(filename=MAIN_RAW_DATA_PATH)
