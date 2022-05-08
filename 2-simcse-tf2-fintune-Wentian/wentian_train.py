import os

os.environ["TF_KERAS"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# from tensorflow.keras.utils import multi_gpu_model

from bert4keras.snippets import sequence_padding
from simcse_tf2.simcse import simcse
from simcse_tf2.data import get_tokenizer, load_data, SimCseDataGenerator
from simcse_tf2.losses import simcse_loss
import tensorflow as tf
import numpy as np



def texts_to_ids(data, tokenizer, max_len=64):
    """转换文本数据为id形式
    """
    token_ids = []
    for d in data:
        token_ids.append(tokenizer.encode(d, maxlen=max_len)[0])
    return sequence_padding(token_ids)


def encode_fun(texts, model, tokenizer, maxlen):
    inputs = texts_to_ids(texts, tokenizer, maxlen)

    embeddings = model.predict([inputs, np.zeros_like(inputs)])
    return embeddings


if __name__ == '__main__':
    # 1. bert config
    model_path = '/home/zqxie/project/Sim_bert_v2/roformer-sim-main/output/res_finetune_mcpr_L_after_25_epoch/best_model/'
    checkpoint_path = '%s/bert_model.ckpt' % model_path
    config_path = '%s/bert_config.json' % model_path
    dict_path = '%s/vocab.txt' % model_path

    # 2. set hyper parameters
    max_len = 70
    pooling = 'first-last-avg'  # in ['first-last-avg', 'last-avg', 'cls', 'pooler']
    dropout_rate = 0.1
    batch_size = 50
    learning_rate = 5e-5
    epochs = 3
    output_units = 128
    activation = 'tanh'

    # 3. data generator
    train_data = load_data('/home/zqxie/project/WenTianSearch-main_ori/data/query_doc.csv', delimiter=",")
    # train_data = load_data('./examples/data/sup_sample.csv', delimiter = "\t")
    train_generator = SimCseDataGenerator(train_data, dict_path, batch_size, max_len)
    # print(next(train_generator.forfit()))

    # 4. build model
    model = simcse(config_path, checkpoint_path, dropout_rate=dropout_rate, output_units=output_units,
                   output_activation=activation)
    # model = multi_gpu_model(model, gpus=2)

    print(model.summary())
    # 5. model compile
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss=simcse_loss, optimizer=optimizer)

    # 6. model fit
    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs)

    import csv
    from tqdm import tqdm

    pre_batch_size = 5000
    corpus = [line[1] for line in csv.reader(open("./examples/data/corpus.tsv"), delimiter='\t')]
    query = [line[1] for line in csv.reader(open("./examples/data/dev.query.txt"), delimiter='\t')]
    tokenizer = get_tokenizer(dict_path)
    query_embedding_file = csv.writer(open('./examples/query_embedding', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), pre_batch_size)):
        batch_text = query[i:i + pre_batch_size]
        print("query size:", len(batch_text))
        temp_embedding = encode_fun(batch_text, model, tokenizer, max_len)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            query_embedding_file.writerow([i + j + 200001, writer_str])
    print("query end!")
    doc_embedding_file = csv.writer(open('./examples/doc_embedding', 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), pre_batch_size)):
        batch_text = corpus[i:i + pre_batch_size]
        temp_embedding = encode_fun(batch_text, model, tokenizer, max_len)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            doc_embedding_file.writerow([i + j + 1, writer_str])
    print("doc end!")
