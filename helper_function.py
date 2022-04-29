from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import numpy as np
import matplotlib as mpl
import pandas as pd
import re
from pythainlp.tokenize import Tokenizer
from pythainlp.ulmfit import process_thai

emoji_regx = '(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])'


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=10):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
        print(label)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

def change_matplotlib_font(font_download_url):
    FONT_PATH = 'MY_FONT'
    
    font_download_cmd = f"wget {font_download_url} -O {FONT_PATH}.zip"
    unzip_cmd = f"unzip -o {FONT_PATH}.zip -d {FONT_PATH}"
    os.system(font_download_cmd)
    os.system(unzip_cmd)
    
    font_files = fm.findSystemFonts(fontpaths=FONT_PATH)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)

    font_name = fm.FontProperties(fname=font_files[0]).get_name()
    mpl.rc('font', family=font_name)
    print("font family: ", plt.rcParams['font.family'])
    # import new mathplotlib that attach thai front

def dummy_fun(doc):
    return doc

def customize_text_tokenizer(reviewText, engine='attacut', split=False):
    _tokenizer = Tokenizer(engine='attacut')
    without_url = re.sub(r"http\S+", "", reviewText)
    # replace 55 with 'ฮ่า' before clean word
    reviewText = re.sub(r"(555)", ' ฮ่า', without_url)
    # Thai & English Charecter preserve 
    pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z']|^'|'$|''|[//t//n]|^#|"+emoji_regx)
    char_to_remove = re.findall(pattern, reviewText)
    # preserve sepperate token -
    char_to_remove = [i for i in char_to_remove if i != '-']
    # remove ignore charecters
    list_with_char_removed = [char for char in reviewText if not char in char_to_remove]
    result = ''.join(list_with_char_removed).strip()
    # tokenize word with attacut
    if engine == 'thai_process':
      result = process_thai(result, tok_func=_tokenizer.word_tokenize)
    else:
      result = word_tokenize(result, engine=engine, keep_whitespace=True)

    return result

font_download_url = "https://fonts.google.com/download?family=Sarabun"
# change_matplotlib_font(font_download_url)

# hand-craft features
def generate_handcraft_features(merge_chat_df):
    '''
    return
        wc - words count
        uwc - unique words count
        processed_chat_len - charecters lenght after preprocessing
        postag - pos tagging a word in messages in dict()
    '''
    wc = merge_chat_df.message.map(lambda row: len([j for i in row for j in i ]))
    uwc = merge_chat_df.message.map(lambda row: len(set([j for i in row for j in i ])))
    processed_chat_len = merge_chat_df.message.map(lambda row: len(''.join([j for i in row for j in i ])))
    postag = merge_chat_df.message.apply(lambda x: [pos_tag(i, corpus='orchid') for i in x]).apply(lambda postag: dict(Counter([j[1] for i in postag for j in i])))
    return wc, uwc, processed_chat_len, postag

def extract_handcraft_feature(merge_chat_df):
    '''
        postag_df : extract pos tagging colum and fill nan value = 0
        timestamp_chat_avg : extract a rapid message sent to bot, to robust feature we max a value at 1800 second(30min)
                            if more than 30 min we will just in another session of conversation
    '''
    wc, uwc, processed_chat_len, postag = generate_handcraft_features(merge_chat_df)
    postag_df = pd.json_normalize(postag).fillna(0)
    merge_chat_df['wc'] = wc
    merge_chat_df['uwc'] = uwc
    merge_chat_df['processed_chat_len'] = processed_chat_len
    merge_chat_df['timestamp_chat_avg'] = merge_chat_df.timestamp_chat.apply(lambda x: np.mean([ 0 if i > (30 * 60) else i for i in x]))
    merge_chat_df['hour_in_datetime'] = merge_chat_df['hour_in_datetime'].apply(lambda x: np.mean(x))
    merge_chat_df = pd.concat([merge_chat_df, postag_df], axis=1)
    return merge_chat_df
