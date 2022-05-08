from tqdm import  tqdm
# with open('data_generated_pairs.txt', 'w') as up:
#     with open('data_generated.tsv', "r", encoding="utf8") as fr:
#         cnt = 0
#         for line in tqdm(fr):
#             line = line.strip().split('\t')
#             if  int(line[-1]) == 1 and len(line)==3:
#                 up.write('{}\t{}\n'.format(line[0], line[1]))
#             # if cnt> 50:
            #     break

with open('/home/zqxie/project/Sim_bert_v2/roformer-sim-main/ecom/train.query.txt', "r", encoding="utf8") as fr:
    t = [line.strip().split("\t") for line in fr]
    print(len(t))
    # for line in fr:
    #     cur = line.strip().split("\t")
    #     if len(cur) == 1:
    #         print(line)
        # break