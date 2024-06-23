import pandas as pd
import numpy as np
from contractions import contractions_dict
import re
import string
import unicodedata
import evaluate


def extract_pandas_df(transcript,description):
    trans_vid_id = []
    trans_desc = []
    with open(transcript,'r', encoding="utf-8") as t:
        A = t.readline()
        while A != "":
            B = A.split()
            trans_vid_id.append(B[0])
            trans_desc.append(" ".join(B[1:]))
            A = t.readline()
    
    desc_vid_id = []
    desc_desc = []
    with open(description,'r', encoding="utf-8") as d:
        C = d.readline()
        while C != "":
            D = C.split()
            desc_vid_id.append(D[0])
            desc_desc.append(" ".join(D[1:]))
            C = d.readline()
            
    df1 = pd.DataFrame({'Video_id':trans_vid_id,"Vid_Transcript":trans_desc})
    df2 = pd.DataFrame({'Video_id':desc_vid_id,"Vid_summary":desc_desc})
    
    df = pd.merge(df1,df2,how = 'inner',on = 'Video_id')
    return df



def expand_contractions(text, contraction_map=contractions_dict):
    # Using regex for getting all contracted words
    contractions_keys = '|'.join(contraction_map.keys())
    contractions_pattern = re.compile(f'({contractions_keys})', flags=re.DOTALL)

    def expand_match(contraction):
        # Getting entire matched sub-string
        match = contraction.group(0)
        expanded_contraction = contraction_map.get(match)
        if not expand_contractions:
            print(match)
            return match
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Remove puncuation from word
def rm_punc_from_word(word):
    clean_alphabet_list = [
        alphabet for alphabet in word if alphabet not in string.punctuation
    ]
    return ''.join(clean_alphabet_list)

# print(rm_punc_from_word('#cool!'))


# Remove puncuation from text
def rm_punc_from_text(text):
    clean_word_list = [rm_punc_from_word(word) for word in text]
    return ''.join(clean_word_list)

# Cleaning text
def clean_text(text):
    text = expand_contractions(text)
    text = text.lower()
    text = rm_punc_from_text(text)
    
    # there are hyphen(–) in many titles, so replacing it with empty str
    # this hyphen(–) is different from normal hyphen(-)
    text = re.sub('–', '', text)
    text = ' '.join(text.split())  # removing `extra` white spaces

    # Removing unnecessary characters from text
    text = re.sub("(\\t)", ' ', str(text)).lower()
    text = re.sub("(\\r)", ' ', str(text)).lower()
    text = re.sub("(\\n)", ' ', str(text)).lower()

    # remove accented chars ('Sómě Áccěntěd těxt' => 'Some Accented text')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode(
        'utf-8', 'ignore'
    )

    text = re.sub("(__+)", ' ', str(text)).lower()
    text = re.sub("(--+)", ' ', str(text)).lower()
    text = re.sub("(~~+)", ' ', str(text)).lower()
    text = re.sub("(\+\++)", ' ', str(text)).lower()
    text = re.sub("(\.\.+)", ' ', str(text)).lower()

    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(text)).lower()

    text = re.sub("(mailto:)", ' ', str(text)).lower()
    text = re.sub(r"(\\x9\d)", ' ', str(text)).lower()
    text = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(text)).lower()
    text = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM',
                  str(text)).lower()

    text = re.sub("(\.\s+)", ' ', str(text)).lower()
    text = re.sub("(\-\s+)", ' ', str(text)).lower()
    text = re.sub("(\:\s+)", ' ', str(text)).lower()
    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()

    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(text))
        repl_url = url.group(3)
        text = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(text))
    except Exception as e:
        pass

    text = re.sub("(\s+)", ' ', str(text)).lower()
    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()

    return text

def clean_dataset(ds):
    ds = ds.map(lambda x: {"Vid_summary": [clean_text(o) for o in x["Vid_summary"]]}, batched=True)

    # Apply the clean_text function to the Vid_Transcript column
    ds = ds.map(lambda x: {"Vid_Transcript": [clean_text(o) for o in x["Vid_Transcript"]]}, batched=True)
    
    return ds


rouge_score = evaluate.load("rouge")