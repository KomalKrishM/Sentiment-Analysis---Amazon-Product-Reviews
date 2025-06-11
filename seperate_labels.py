import pandas as pd
import os

def clean_labelled_text_file(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Converts FastText-style labelled text (__label__1 or __label__2) into a cleaned DataFrame.

    Args:
        input_file (str): Path to the raw .txt file.
        output_file (str, optional): If provided, saves the cleaned data to a CSV file.

    Returns:
        pd.DataFrame: A DataFrame with 'text' and 'label' columns.
    """
    texts = []
    labels = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("__label__1"):
                labels.append(0)
                texts.append(line.replace("__label__1", "").strip())
            elif line.startswith("__label__2"):
                labels.append(1)
                texts.append(line.replace("__label__2", "").strip())

    # print(f"Labels: {len(labels)}")
    # print(f"Texts: {len(texts)}")
    # pr

    df = pd.DataFrame({'text': texts, 'label': labels})

    if output_file:
        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the file
        df.to_csv(output_file, index=False)
        print(f"✅ Saved cleaned dataset with {len(df)} samples to: {output_file}")

    return df

df = clean_labelled_text_file("/Users/komalkrishnamogilipalepu/Downloads/archive/train.ft.txt", "/Users/komalkrishnamogilipalepu/Downloads/archive/train_dataset.csv")
print(df.head())


