
import json
import random

# Synonym dictionary for data augmentation
SYNONYM_DICT = {
    "맞춤법": ["표기법", "철자법", "쓰기 규칙"],
    "띄어쓰기": ["띄어 씀", "공백 규정"],
    "표준어": ["표준 발음", "공식 언어"],
}

# Function to augment data by replacing synonyms
def augment_with_synonyms(data):
    augmented_data = []
    for item in data:
        original_question = item["input"]["question"]
        found_synonym = False
        for keyword, synonyms in SYNONYM_DICT.items():
            if keyword in original_question:
                # Add the original item
                if item not in augmented_data:
                    augmented_data.append(item)
                # Add augmented items
                for synonym in synonyms:
                    new_question = original_question.replace(keyword, synonym)
                    new_item = item.copy()
                    new_item["input"]["question"] = new_question
                    augmented_data.append(new_item)
                found_synonym = True
                break
        if not found_synonym:
            augmented_data.append(item)
    return augmented_data

# Main function to run the augmentation
if __name__ == "__main__":
    # Load the original training data
    with open("korean_language_rag_V1.0_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Augment the data
    augmented_train_data = augment_with_synonyms(train_data)

    # Save the augmented data to a new file
    with open("korean_language_rag_V1.0_train_augmented.json", "w", encoding="utf-8") as f:
        json.dump(augmented_train_data, f, ensure_ascii=False, indent=4)

    print(f"Original data count: {len(train_data)}")
    print(f"Augmented data count: {len(augmented_train_data)}")
    print("Augmented data saved to korean_language_rag_V1.0_train_augmented.json")
