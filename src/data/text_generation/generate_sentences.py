import random
import warnings

import pandas as pd
import torch
from sqlalchemy import create_engine
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

warnings.filterwarnings("ignore")


def random_state(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


random_state(1234)

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
        torch_device)
    translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences,
                                temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


template_phrases = [
    "CMP1 has acquired CMP2 <filler>",
    'CMP1 announced its intention to acquire CMP2 <filler>',
    'CMP1 acquired CMP2 <filler>',
    'CMP1 to acquire CMP2 <filler>'
    'CMP1 to buy CMP2 <filler>',
    'CMP1 closes deal to buy CMP2 <filler>',
    'CMP1 completed its acquisition of CMP2 <filler>',
    'CMP2 was acquired by CMP1 <filler>',
    'CMP2 has been acquired by CMP1 <filler>',
    'CMP1 merges with CMP2 <filler>',
    'CMP1 and CMP2 have begun their merger <filler>',
]

filler = [
    'today',
    'yesterday',
    'recently',
    'in a recent business move',
    'in an attempt to improve service',
    'this week',
    'this month',
    'in $<decimal> billion deal',
    'in $<decimal> bn deal',
    'in $<integer> million deal',
    'in $<integer> mn deal',
    'for a sum total of $<decimal> billion',
    'for a value of $<decimal> billion',
    'for a sum total of $<integer> million',
    'for a value of $<integer> million',
    'for $<integer> million',
    'for $<decimal> billion',
    'as part of a new corporate plan',
    ''
]

terminators = [
    'llc',
    'limited',
    'ltd',
    'inc',
    'corp',
    'incorporated',
    'co'
]

company_name_file = open('training_company_names.txt')
company_names = [line.rstrip() for line in company_name_file]
company_name_file.close()

all_phrases = set()


def process_paraphrase(paraphrase_in: str):
    if 'CMP1' not in paraphrase_in or 'CMP2' not in paraphrase_in:
        return
    cmp1 = random.choice(company_names)
    if random.randint(0, 3) == 0:
        cmp1 += ' ' + random.choice(terminators)
    cmp2 = cmp1
    while cmp2 == cmp1:
        cmp2 = random.choice(company_names)
    if random.randint(0, 3) == 0:
        cmp2 += ' ' + random.choice(terminators)
    processed_paraphrase = paraphrase_in.replace('CMP1', cmp1).replace('CMP2', cmp2)
    all_phrases.add((processed_paraphrase, cmp2))


def generate_phrases():
    for raw_phrase in template_phrases:
        for i in range(5):
            phrase = raw_phrase.replace('<filler>', random.choice(filler))
            phrase = phrase.replace('<decimal>', str(random.randint(0, 100) / 10))
            phrase = phrase.replace('<integer>', str(random.randint(0, 100)))

            # print("-" * 100)
            # print("Input_phrase: ", phrase)
            # print("-" * 100)
            responses = get_response(phrase, 10, 10)
            for paraphrase in responses:
                process_paraphrase(paraphrase)

    list_phrases = list(all_phrases)
    random.shuffle(list_phrases)
    return list_phrases


phrase_df = pd.DataFrame(
    {},
    columns=["text", "isMerger", "isAcquisition", "isMergerOrAcquisition", "website"],
)

phrases = generate_phrases()
for phrase in phrases:
    phrase_df = phrase_df.append(
        {
            "text": phrase[0],
            "isMerger": float("NaN"),
            "isAcquisition": float("NaN"),
            "isMergerOrAcquisition": 1,
            "website": 'RANDOMLY_GENERATED_FAKE_DATA',
            "company": phrase[1],
        },
        ignore_index=True,
    )

print(phrase_df)
engine = create_engine(
    "sqlite:///../../../data/processed/test_train_database.db", echo=True
)
sqlite_connection = engine.connect()
sentence_table_name = "DryCleaned"
phrase_df.to_sql(
    sentence_table_name, sqlite_connection, if_exists="append", index=False
)
