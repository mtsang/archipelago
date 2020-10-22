all_sent_file = "glue_data/SST-2/original/datasetSentences.txt"
train_file = "glue_data/SST-2/train_all.tsv"

f1 = open(all_sent_file)
f2 = open(train_file)
fw = open("glue_data/SST-2/train.tsv", "w")

lines1, lines2 = f1.readlines(), f2.readlines()
fw.write(lines1[0])

hash_set = set()
for line in lines1:
    sent = line.split("\t")[-1]
    hash_set.add(sent.strip().lower())

for line in lines2:
    sent = line.split("\t")[0]
    if sent.strip() in hash_set:
        fw.write(line)
