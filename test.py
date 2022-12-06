from custom_vectorizer import CustomTfidForTokenizedfVectorizer, CustomVectorizer
import pandas as pd

def main():

    dirpath = 'dataset/'
    depression_questionaire_df = pd.read_csv(dirpath+'depression_project_datasetV1_depression_questionair.csv', sep=",")
    depression_questionaire_df.columns = ["customer_id","condition","score","result","timestamp","chat_count"]
    depression_chat_df = pd.read_csv(dirpath+'depression_project_datasetV1_depression_chat.csv', sep=",")
    depression_chat_df.columns = ["customer_id","message","timestamp"]
    depression_questionaire_df['condition'] = depression_questionaire_df['condition'].replace(['{{de_high}}', '{{de_serious}}'], ['{{de_serious}}', '{{de_high}}'])
    tffidf = CustomTfidForTokenizedfVectorizer(ngram_range=(1,2), use_idf=True)


main()
