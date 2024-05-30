import argparse
import text
from utils import load_filepaths_and_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument(
        "--filelists",
        nargs="+",
        default=[
            "filelists/ljs_audio_text_val_filelist.txt",
            "filelists/ljs_audio_text_test_filelist.txt",
        ],
    )
    parser.add_argument("--text_cleaners", nargs="+", default=["korean_cleaners"])

    args = parser.parse_args()

    
for filelist in args.filelists:
    print("START:", filelist)
    with open(filelist, "r", encoding="utf-8") as f:
        lines = f.readlines()

    batch_size = 64
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]

        for line in batch_lines:
            parts = line.strip().split("|")
            original_text = parts[args.text_index]
            cleaned_text = text._clean_text(original_text, args.text_cleaners)
            parts[args.text_index] = cleaned_text

            new_filelist = filelist + "." + args.out_extension
            with open(new_filelist, "a", encoding="utf-8") as f:
                f.write("|".join(parts) + "\n") 
