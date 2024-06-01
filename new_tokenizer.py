import pandas as pd
import new_preprocessing as preprocessing
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_pickle("data/data_training.pkl")

# filter label based on target pos in labels array, clean/preprocess and return dataframe
def filter_and_tokenize(target_df, target_pos_array) -> pd.DataFrame:
    # create empty dict for appending values
    filtered_dict = {
        "work_id": [],
        "labels": [],
        "text": [],
        "clean_text": [],
        "tokenized_text": [],
        "proper_nouns": []  # Add a new column for proper nouns
    }

    # loop through df and append rows when target pos matches label value
    for i, row in target_df.iterrows():
        valid = False
        for target_pos in target_pos_array:
            if row["labels"][target_pos] == 1:
                valid = True
        if valid:
            filtered_dict["work_id"].append(row["work_id"])
            filtered_dict["labels"].append(row["labels"])
            filtered_dict["text"].append(row["text"])
            clean_text = preprocessing.preprocess_text(row["text"])
            
            # extract and exclude proper nouns from the text
            proper_nouns = []
            for token in nlp(clean_text):
                if token.pos_ == "PROPN":
                    proper_nouns.append(token.lemma_)

            for proper_noun in proper_nouns:
                clean_text = clean_text.replace(proper_noun, "")
            
            # remove any leftover whitespace
            clean_text = ' '.join(clean_text.split())
            
            # append the clean text to the dictionary
            filtered_dict["clean_text"].append(clean_text)
            
            # Tokenize and lemmatize the cleaned text
            tokenized_text = preprocessing.tokenize_and_lemmatize(clean_text)
            filtered_dict["tokenized_text"].append(tokenized_text)

            # append the extracted proper nouns to the dictionary
            filtered_dict["proper_nouns"].append(proper_nouns)

    return pd.DataFrame(filtered_dict)


if __name__ == "__main__":
    miscarriage = filter_and_tokenize(df, [7, 20, 23, 25])
    war_trauma = filter_and_tokenize(df, [1, 5, 2, 13, 6, 14])
    eating_disorders = filter_and_tokenize(df, [17, 19, 26])
    
    miscarriage.to_csv("traumas/*.csv")
    war_trauma.to_csv("traumas/*.csv")
    eating_disorders.to_csv("traumas/*.csv")    
