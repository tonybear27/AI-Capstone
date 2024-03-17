import pandas as pd
from googletrans import Translator

def translate(text):
    translator = Translator()
    translation = translator.translate(text, dest='fr')
    print(translation.text)
    return translation.text

df = pd.read_csv('large.csv')
df_translated = df.copy()
df_translated['Title'] = df['Title'].apply(translate)

df_translated.to_csv('French.csv', index=False)