import os
import json
import random

random.seed(42)

INPUT_ROOT_DIR = "../dataset/VL" 
OUTPUT_TRAIN_FILE = "filelists/train.txt"
OUTPUT_VAL_FILE = "filelists/val.txt"
OUTPUT_TEST_FILE = "filelists/test.txt"

file_list = []

speaker_files = {}
for root, dirs, files in os.walk(INPUT_ROOT_DIR):
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(root, file)
            with open(json_path, "r", encoding="utf-8") as f_json:
                data = json.load(f_json)
                if "speaker" in data:
                    speaker_id = data["speaker"]["id"]
                    if speaker_id not in speaker_files:
                        speaker_files[speaker_id] = []
                    speaker_files[speaker_id].append((json_path, data["script"]["text"]))

for speaker_id, files in speaker_files.items():
    random.shuffle(files)
    file_list.extend(files)


random.shuffle(file_list)
num_files = len(file_list)
num_train = int(num_files * 0.8)
num_val = int(num_files * 0.1)
num_test = num_files - num_train - num_val

train_files = file_list[:num_train]
val_files = file_list[num_train:num_train + num_val]
test_files = file_list[num_train + num_val:]

def write_file_list(file_list, output_file):
    with open(output_file, "w", encoding="utf-8") as f_out:
        for file_info in file_list:
            json_path, script_text = file_info
            wav_path = json_path.replace(".json", ".wav")
            f_out.write(f"../dataset/VS{wav_path[13:]}\n")

write_file_list(train_files, OUTPUT_TRAIN_FILE)
write_file_list(val_files, OUTPUT_VAL_FILE)
write_file_list(test_files, OUTPUT_TEST_FILE)
