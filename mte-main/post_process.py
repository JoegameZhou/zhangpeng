with open("/code/mte/output_eval.txt", "r") as output1:
    with open("/code/mte/data/iwslt14.de-en.bpe10000/tmp/test.en", "r") as output2:
        with open("/code/mte/processed_output_eval.txt", "a") as processed1:
            with open("/code/mte/processed_test.en", "a") as processed2:
                lines1 = output1.readlines()
                lines2 = output2.readlines()
                max_len = 0
                for i in range(len(lines1)):
                    #print(i, len(lines2[i].split(' ')))
                    if len(lines2[i].split(' ')) > max_len:
                        max_len = len(lines2[i].split(' '))
                    #if len(lines1[i]) != 1 and len(lines2[i].split(' ')) <= 128 and len(lines2[i].split(' '))/len(lines1[i].split(' ')) <= 2:
                    if len(lines1[i]) != 1:
                        processed1.write(lines1[i])
                        processed2.write(lines2[i])
                print(max_len)