import json

# the file to be converted to
# json format
filename = "/Data/winter_olympics.txt"

# dictionary where the lines from
# text will be stored
dict1 = {}

# creating dictionary
with open(filename) as fh:
    for line in fh:
        # reads each line and trims of extra the spaces
        # and gives only the valid words
        command, description = line.strip().split(None, 1)

        dict1[command] = description.strip()

    # creating json file
# the JSON file is named as test1
out_file = open("test1.json", "w")
json.dump(dict1, out_file, indent=4, sort_keys=False)
out_file.close()