from translator import TextTranslator
import pandas as pd 

def main():
    # Initialize the translator
    translator = TextTranslator()
    input_file = "1k review angah.csv"
    output_file = "Re.csv"
    translator.translate_csv_to_english(input_file, output_file)
    print(f"Excel file translated and saved to {output_file}")
    
    df_translate = pd.read_csv(output_file)
    print(df_translate.head())
if __name__ == "__main__":
    main()