# translator.py

from googletrans import Translator
import pandas as pd
from tqdm import tqdm  # tqdm for progress bar

class TextTranslator:
    def __init__(self):
        self.translator = Translator()

    def translate_to_english(self, text):
        try:
            translation = self.translator.translate(text, dest='en')
            return translation.text
        except Exception as e:
            print(f"Translation failed for: {text}")
            print(f"Error: {e}")
            return None  # Return None for failed translations

    def translate_csv_to_english(self, input_file, output_file, batch_size=10):
        df = pd.read_csv(input_file)
        
        # Translate in batches to avoid potential issues with rate limiting
        tqdm.pandas()  # Enable progress bar for pandas
        df['English Review'] = df['Review'].progress_apply(self.translate_to_english)
        
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    translator = TextTranslator()

    input_file = "REVIEW LIST.csv"
    output_file = "TranslatedData.csv"

    translator.translate_csv_to_english(input_file, output_file)
    print(f"CSV file translated and saved to {output_file}")
