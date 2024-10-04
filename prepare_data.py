import pandas as pd
import numpy as np
import os
import shutil
import random
from tqdm import tqdm
from datasets import Dataset, Image, Sequence, load_dataset

file_path = '/path/to/labels.xlsx'

df = pd.read_excel(file_path, sheet_name=0)
print('Combined data shape:', df.shape)
print('Missing values:', pd.isna(df).sum())


def sanitize(value):
    return value.replace(' ', '_').replace('(', '').replace(')', '').lower()


df['id'] = df['id'].astype(str)
df['id'] = df['id'].apply(sanitize)


root_dataset_dir = 'dataset'
images_dir = os.path.join(root_dataset_dir, 'images')
os.makedirs(images_dir, exist_ok=True)


source_dir = '/path/to/images'

for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.jpg'):
            src_file_path = os.path.join(root, file)
            dst_file_path = sanitize(os.path.join(images_dir, file))
            shutil.copy(src_file_path, dst_file_path)
            print(f"Copied: {src_file_path} to {dst_file_path}")


print('Checking missing ids...')
missing_id = []
for i in os.listdir(images_dir):
    file_id = i[:-4]
    if file_id not in df['id'].values:
        missing_id.append(i)
print('Number of missing ids:', len(missing_id))

print('Checking missing images...')
missing_images = []
for i in df['id'].values:
    i = str(i)
    file_id = i+".jpg"
    if file_id not in os.listdir(images_dir):
        missing_images.append(i)
print('Number of missing images:', len(missing_images))

df_unique = df.drop_duplicates(subset=['human'])
print(f"Number of rows after removing duplicates: {len(df_unique)}")
print(f"Number of duplicate human rows: {len(df)-len(df_unique)}")

df_unique = df.drop_duplicates(subset=['gpt'])
print(f"Number of rows after removing duplicates: {len(df_unique)}")
print(f"Number of duplicate gpt rows: {len(df)-len(df_unique)}")

# Create test and train/val sets, preserving proportions mutually exclusive unique values
print('Test set proportion:', 0.25)
num_test_values = int(0.25 * len(df_unique))

np.random.seed(123)
test_gpt_values = np.random.choice(df_unique['gpt'], num_test_values, replace=False)

# Create the test and train sets based on the selected 'gpt' values
df_test = df[df['gpt'].isin(test_gpt_values)]
df_train_val = df[~df['gpt'].isin(test_gpt_values)]

# Check the sizes of the resulting dataframes
print(f"Number of rows in the test set: {len(df_test)}")
print(f"Number of rows in the training set: {len(df_train_val)}")
print(f"Number of unique 'gpt' values in the test set: {df_test['gpt'].nunique()}")
print(f"Number of unique 'gpt' values in the training set: {df_train_val['gpt'].nunique()}")

df_train_val.to_excel("train_and_val.xlsx", index=True)
df_test.to_excel("test.xlsx", index=True)


# Process the data into JSON format
def row_to_dict(row):
    return {
        "id": row['id'],
        "image": f"{row['id']}.jpg",
        "conversations": [
            {
                "from": "human",
                "value": row['human']
            },
            {
                "from": "gpt",
                "value": row['gpt']
            }
        ]
    }


json_data = df_train_val.apply(row_to_dict, axis=1).tolist()

# Shuffle the data
data = json_data.copy()
random.shuffle(data)

# Transform the data into the format expected by HuggingFace
transformed_data = []

for entry in data:
    image_file = os.path.join(images_dir, entry['image'])
    conversations = entry['conversations']

    conversation_list = []

    # transform each conversation pair
    for i in range(0, len(conversations), 2):
        user_message = {
            "content": [
                {"index": None, "text": conversations[i]['value'], "type": "text"},
                {"index": 0, "text": None, "type": "image"}
            ],
            "role": "user"
        }

        assistant_message = {
            "content": [
                {"index": None, "text": conversations[i+1]['value'], "type": "text"}
            ],
            "role": "assistant"
        }

        conversation_list.append(user_message)
        conversation_list.append(assistant_message)

    transformed_data.append({
        "messages": conversation_list,
        "images": image_file
    })

dataset_dict = {
    "messages": [entry["messages"] for entry in transformed_data],
    "images": [[entry["images"]] for entry in transformed_data]
}

ds = Dataset.from_dict(dataset_dict)
ds = ds.cast_column("images", Sequence(Image()))
ds_split = ds.train_test_split(test_size=0.2)

ds_split.save_to_disk("dataset/hf")


# Check dataset is consistent and correct format
def check_consistency(dataset1, dataset2, num_samples):
    # Check split names and non-emptiness
    if list(dataset1.keys()) != list(dataset2.keys()):
        print("Error: Dataset splits do not match.")
        return False

    for split_name in dataset1.keys():
        if len(dataset1[split_name]) == 0 or len(dataset2[split_name]) == 0:
            print(f"Error: Split '{split_name}' is empty in one or both datasets.")
            return False

    # Check column names consistency
    for split_name in dataset1.keys():
        if list(dataset1[split_name].column_names) != list(dataset2[split_name].column_names):
            print(f"Error: Column names in split '{split_name}' do not match.")
            return False

    # Check type/format consistency for the first 20 records
    for split_name in dataset1.keys():
        num_records_to_check = min(len(dataset1[split_name]), len(dataset2[split_name]), num_samples)
        for i in tqdm(range(num_records_to_check), desc=f"Checking split '{split_name}'"):
            record1 = dataset1[split_name][i]
            record2 = dataset2[split_name][i]
            for column in dataset1[split_name].column_names:
                if type(record1[column]) != type(record2[column]):
                    print(f"Error: Type mismatch for column '{column}' in split '{split_name}', records {i} in dataset1 and dataset2.")
                    return False

    # If all checks pass
    print("Dataset consistency checks passed.")
    return True


llava_instruct_ds = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")

is_valid = check_consistency(ds_split.copy(), llava_instruct_ds.copy(), num_samples=ds_split.num_rows['test'])
print(f"Dataset is valid: {is_valid}")

print("First sample of reference dataset")
print(llava_instruct_ds["train"][0])

print("First sample of the processed Ubitus dataset")
print(ds_split["train"][0])
