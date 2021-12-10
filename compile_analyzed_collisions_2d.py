
import os
import json
import glob

def main():
    files = glob.iglob(f"analyzed_collisions/**/*.json", recursive=True)
    
    dicts = []

    for file in files:
        with open(file, "r") as f:
            dicts.append(json.load(f))

    header = dicts[0].keys()
    rows = []

    for d in dicts:
        rows.append(d.values())

    data = [header] + rows

    with open("analyzed_collisions.tsv", "w") as f:
        for row in data:
            str_row = [str(x) for x in row]
            f.write("\t".join(str_row) + "\n")




if __name__ == '__main__':
    main()
