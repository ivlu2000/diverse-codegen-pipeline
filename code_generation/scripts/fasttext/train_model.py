import os
import fasttext
import pandas as pd
from datasets import load_dataset


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


dataset = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)
# split the dataset into two parts
df = pd.read_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))

df["label"] = "positive"

negative_samples = []
# Create negative examples by sampling randomly from IMDB dataset
negative_samples_count = len(df)
index = 0

for sample in dataset:
    if index < negative_samples_count:
        negative_samples.append(sample["text"])
        index += 1
    else:
        break

random_docs = pd.DataFrame(negative_samples, columns=["task"])
random_docs["label"] = "negative"

# Combine positive and negative samples
combined_df = pd.concat([df, random_docs], ignore_index=True)
# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape: {combined_df.shape}")
print(f"Positive samples: {len(combined_df[combined_df['label'] == 'positive'])}")
print(f"Negative samples: {len(combined_df[combined_df['label'] == 'negative'])}")

os.makedirs(os.path.join(root_dir, "datasets/fasttext"), exist_ok=True)

# create empty file
open(os.path.join(root_dir, "datasets/fasttext/fasttext_train.txt"), "w").close()

# Create a text file in the format FastText expects
fasttext_file_path = os.path.join(root_dir, "datasets/fasttext/fasttext_train.txt")
with open(fasttext_file_path, "w", encoding="utf-8") as f:
    for _, row in combined_df.iterrows():
        # Format: __label__LABEL TEXT
        f.write(f"__label__{row['label']} {row['task']}\n")

print(f"FastText training file created at: {fasttext_file_path}")

# Train the model using the properly formatted text file
model = fasttext.train_supervised(
    input=fasttext_file_path,
    dim=256,
    lr=0.1,
    wordNgrams=3,
    minCount=3,
    epoch=20,
)

os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)
# Save the model
model.save_model(os.path.join(root_dir, "models/fasttext_model.bin"))

# Test the model on sample sentences
test_sentences = [
    "Create a function to calculate the factorial of a number",
    "Write a program to find the largest number in an array",
    "I loved this movie, it was amazing and the actors were great",
    "The food at this restaurant was terrible and the service was slow",
    "USS Congress was a nominally rated 38-gun wooden-hulled, three-masted heavy frigate launched on 15 August 1799. She was one of the original six frigates of the newly formed United States Navy and, along with her sister ships, was larger and more heavily armed than standard frigates of the period. Her first duties were to protect American shipping during the Quasi-War with France. In 1804 and 1805, Congress helped to defeat the Barbary corsairs in the First Barbary War. During the War of 1812, she made several extended cruises with President: the pair captured 20 British merchant ships. At the end of 1813, due to a lack of materials to repair her, Congress was placed in reserve. In 1815, she took part in the Second Barbary War and made patrols through 1816. In the 1820s, she helped suppress piracy in the West Indies, made several voyages to South America, and was the first U.S. warship to visit China. Congress spent her last ten years as a receiving ship until she was broken up in 1834.",
    "Apollo 9 was the third crewed mission in the United States Apollo program. Launched by a Saturn V rocket from the Kennedy Space Center on March 3, 1969, and flown in low Earth orbit, the mission flight-qualified the Lunar Module (LM), showing that its crew could fly it independently, then rendezvous and dock, as would be required for Apollo 11, the first crewed lunar landing. Commander James McDivitt, Command Module Pilot David Scott, and Lunar Module Pilot Rusty Schweickart tested systems and procedures critical to landing on the Moon. A spacewalk tested the extravehicular life support backpack. McDivitt and Schweickart, entering the LM through the docking tunnel, became the first humans to pass between spacecraft without going outside them, two months after Soviet cosmonauts spacewalked to transfer between Soyuz 4 and Soyuz 5. Apollo 9, a complete success, landed in the Atlantic Ocean on March 13 and was followed by Apollo 10, the dress rehearsal for Apollo 11. This photograph, taken by Schweickart, shows Scott performing a stand-up extravehicular activity from the Command Module Gumdrop, seen from the docked LM Spider with the Earth in the background.",
    "Dolphins Ask Williams for  #36;8.6 Million (AP) AP - The Miami Dolphins have asked Ricky Williams to return  #36;8.6 million they say the running back owes the team because he has decided to retire.",
]

for sentence in test_sentences:
    prediction = model.predict(sentence)
    print(f"Text: {sentence}")
    print(f"Predicted label: {prediction[0][0].replace('__label__', '')}")
    print(f"Confidence: {prediction[1][0]:.4f}")
    print()
