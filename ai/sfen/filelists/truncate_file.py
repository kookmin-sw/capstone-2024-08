def truncate_file(file_path, num_lines):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    truncated_lines = lines[:num_lines]

    with open(file_path, 'w') as file:
        file.writelines(truncated_lines)

truncate_file('train.txt', 1000)
truncate_file('test.txt', 20)
truncate_file('val.txt', 200)
