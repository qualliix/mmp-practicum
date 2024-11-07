def get_new_dictionary(input_dict_name, output_dict_name):
    dic = dict()
    with open(input_dict_name, "r") as fin:
        fin.readline()
        for line in fin.readlines():
            line = line.replace('-', ' ')
            line = line.replace(',', ' ')
            ar = line.split()
            for word in ar[1:]:
                if (word in dic):
                    dic[word].append(ar[0])
                else:
                    dic[word] = [ar[0]]

    with open(output_dict_name, "w") as fout:
        fout.write(str(len(dic)) + "\n")
        for dragon_word in sorted(dic.keys()):
            fout.write(dragon_word + " - ")
            fout.write(", ".join(sorted(dic[dragon_word])) + "\n")
    return
