from tqdm import tqdm
import json
import torch
import os
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

#
# from transformers import BertConfig, BertModel, AdamW
# from transformers import BertTokenizer
# model_path = '../bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model_config = BertConfig.from_pretrained(model_path)
# label_model = BertModel.from_pretrained(model_path, config=model_config).to('cpu')

def Data_Loader(args,tokenizer):
    #判断是疾病还是证型
    if args.syndrome_diag == 'syndrome':
        #找到所有的标签
        syndromes = []
        with open('{}/cardiovascular.json'.format(args.data_path), 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每行内容为JSON对象，并添加到列表中
                syndromes.append(line.replace('\n', ''))
    if args.syndrome_diag == 'diag':
        syndromes = []
        with open('{}/cardiovascular_diag.json'.format(args.data_path), 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每行内容为JSON对象，并添加到列表中
                syndromes.append(line.replace('\n', ''))

    id2syndrome_dict = {}
    syndrome2id_dict = {}

    y = 0
    for i in range(len(syndromes)):
        id2syndrome_dict[i] = syndromes[i]
        syndrome2id_dict[syndromes[i]] = i

    def get_InputTensor(path):

        contents = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                contents.append(data)

        labels = []
        input_ids = []
        attention_masks = []
        true_splitNumbers = []
        sentences = []


        for content in tqdm(contents, desc='Loading data',total=len(contents)):
            if content[-1] == '':
                continue
            if args.syndrome_diag == 'syndrome':
                sentence = content[10] + '[SEP]' + content[16]
                sentence = sentence.replace('中医望闻切诊：','').replace('表情自然',' ').replace(',','').replace('无异常气味','').replace('形体正常','').replace('语气清','').replace('气息平','').replace('面色红润','')
            if args.syndrome_diag == 'diag':
                sentence = content[8] + '[SEP]' + content[9] + '[SEP]' + content[11] + '[SEP]' + content[13] + '[SEP]' + content[12]
            sentence = sentence.replace('，', '').replace('。', '').replace('、', '').replace('；', '').replace('：',
                                                                                                            '').replace(
                '？', '').replace('！', '').replace('\t', '').replace('主诉', '').replace('现病史', '').replace('*', '').replace(' ', '').replace('“', '').replace('表格<诊断>内容','').replace('\n', '')
            sentences.append(sentence)
            labele_sentence = [0]*args.n_labels
            if args.syndrome_diag == 'syndrome':
                for label in content[-1].split('|'):
                    id = syndrome2id_dict[label]
                    labele_sentence[id] = 1
            if args.syndrome_diag == 'diag':
                for label in content[-2].split('|'):
                    id = syndrome2id_dict[label]
                    labele_sentence[id] = 1
            labels.append(torch.tensor(labele_sentence))
        input_ids_sen = []
        attention_masks_sen = []
        for sentence in tqdm(sentences,desc='Loading sentence vec'):
            sentencei = sentence[:args.max_length]
            encoded_dicti = tokenizer(
                sentencei,  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=args.max_length,  # 填充 & 截断长度
                padding='max_length',
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                truncation=True
            )
            input_idsi = encoded_dicti['input_ids'][0].reshape(1,-1)
            attention_maski = encoded_dicti['attention_mask'][0].reshape(1,-1)
            input_ids_sen.append(input_idsi)
            attention_masks_sen.append(attention_maski)
        input_ids = torch.cat(input_ids_sen, dim=0)
        attention_masks = torch.cat(attention_masks_sen, dim=0)
        labels = torch.stack(labels, dim=0)
        return input_ids, attention_masks, labels

    big = args.big

    if big == 'True':
        input_ids_train, attention_masks_train, labels_train = get_InputTensor(
            '{}/train.json'.format(args.data_path))
        input_ids_test, attention_masks_test, labels_test = get_InputTensor(
            '{}/test.json'.format(args.data_path))
        input_ids_val, attention_masks_val, labels_val = get_InputTensor(
            '{}/dev.json'.format(args.data_path))
    if big == 'False':
        input_ids_train, attention_masks_train, labels_train = get_InputTensor(
            '../data_preprocess/7分类/sne_data/train.json')
        input_ids_test, attention_masks_test, labels_test = get_InputTensor(
            '../data_preprocess/7分类/sne_data/test.json')
        input_ids_val, attention_masks_val, labels_val = get_InputTensor(
            '../data_preprocess/7分类/sne_data/dev.json')
    if big == 'small':
        input_ids_train, attention_masks_train, labels_train = get_InputTensor(
            '../data_preprocess/syndrome_diag/train_small.json')
        input_ids_test, attention_masks_test, labels_test = get_InputTensor(
            '../data_preprocess/syndrome_diag/test_small.json')
        input_ids_val, attention_masks_val, labels_val = get_InputTensor(
            '../data_preprocess/syndrome_diag/dev_small.json')

        # 将输入数据合并为 TensorDataset 对象
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  # 训练样本
        sampler=RandomSampler(train_dataset),  # 随机小批量
        batch_size=args.batch_size,  # 以小批量进行训练
        drop_last=True,
    )

    # 测试集不需要随机化，这里顺序读取就好
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,  # 验证样本
        sampler=SequentialSampler(test_dataset),  # 顺序选取小批量
        batch_size=args.batch_size,
        drop_last=True,
    )

    # 验证集不需要随机化，这里顺序读取就好
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,  # 验证样本
        sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
        batch_size=args.batch_size,
        drop_last=True
    )

    # 用于存储加载的张量的列表
    all_tensors = []
    # 逐个加载文件中的张量并添加到列表中
    for file_path in range(args.n_labels):
        if args.syndrome_diag == 'syndrome':
            loaded_tensor = torch.load('{}/def_vec/label_feature{}.pt'.format(args.data_path,file_path))
            print('{}/def_vec/label_feature{}.pt'.format(args.data_path,file_path))
        # if args.syndrome_diag == 'diag':
        #     loaded_tensor = torch.load('./def_vec/diag/{}/label_feature{}.pt'.format(args.model,file_path))
        #     print('./def_vec/diag/{}/label_feature{}.pt'.format(args.model,file_path))
        all_tensors.append(loaded_tensor)
    if args.model == 'bert':
        hidden_model = 1024
    if args.model == 'longformer':
        hidden_model = 768
    label_feature = torch.stack(all_tensors, dim=0).to(args.device).view(-1, args.define_length, hidden_model)

    return train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict, label_feature

def TCM_SD_Data_Loader(args,tokenizer):
    #找到所有的标签
    syndromes = []
    with open('../data_preprocess/tcm_sd/syndrome_vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每行内容为JSON对象，并添加到列表中
            syndromes.append(line.replace('\n', ''))

    id2syndrome_dict = {}
    syndrome2id_dict = {}

    y = 0
    for i in range(len(syndromes)):
        id2syndrome_dict[i] = syndromes[i]
        syndrome2id_dict[syndromes[i]] = i

    def get_InputTensor(path):

        contents = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                contents.append(data)

        labels = []
        input_ids = []
        attention_masks = []
        true_splitNumbers = []
        sentences = []

        for content in tqdm(contents, desc='Loading data',total=len(contents)):
            if content['norm_syndrome'] == '':
                continue
            sentence = content['chief_complaint'] + '[SEP]' + content['description'] + '[SEP]' + content['detection']
            sentence = sentence.replace('，', '').replace('。', '').replace('、', '').replace('；', '').replace('：',
                                                                                                            '').replace(
                '？', '').replace('！', '').replace('\t', '').replace('主诉', '').replace('现病史', '')
            sentences.append(sentence)
            labele_sentence = [0]*args.n_labels
            for label in content['norm_syndrome'].split('|')[0]:
                id = syndrome2id_dict[label]
                labele_sentence[id] = 1
            labels.append(torch.tensor(labele_sentence))
        input_ids_sen = []
        attention_masks_sen = []
        for sentence in tqdm(sentences,desc='Loading sentence vec'):
            sentencei = sentence[:args.max_length]
            encoded_dicti = tokenizer(
                sentencei,  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=args.max_length,  # 填充 & 截断长度
                padding='max_length',
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                truncation=True
            )
            input_idsi = encoded_dicti['input_ids'][0].reshape(1,-1)
            attention_maski = encoded_dicti['attention_mask'][0].reshape(1,-1)
            input_ids_sen.append(input_idsi)
            attention_masks_sen.append(attention_maski)
        input_ids = torch.cat(input_ids_sen, dim=0)
        attention_masks = torch.cat(attention_masks_sen, dim=0)
        labels = torch.stack(labels, dim=0)
        return input_ids, attention_masks, labels

    big = args.big

    if big == True:
        input_ids_train, attention_masks_train, labels_train = get_InputTensor(
            '../data_preprocess/tcm_sd/train.json')
        input_ids_test, attention_masks_test, labels_test = get_InputTensor(
            '../data_preprocess/tcm_sd/test.json')
        input_ids_val, attention_masks_val, labels_val = get_InputTensor(
            '../data_preprocess/tcm_sd/dev.json')
    if big == False:
        input_ids_train, attention_masks_train, labels_train = get_InputTensor(
            '../data_preprocess/tcm_sd/train_small.json')
        input_ids_test, attention_masks_test, labels_test = get_InputTensor(
            '../data_preprocess/tcm_sd/test_small.json')
        input_ids_val, attention_masks_val, labels_val = get_InputTensor(
            '../data_preprocess/tcm_sd/dev_small.json')

        # 将输入数据合并为 TensorDataset 对象
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  # 训练样本
        sampler=RandomSampler(train_dataset),  # 随机小批量
        batch_size=args.batch_size,  # 以小批量进行训练
        drop_last=True,
    )

    # 测试集不需要随机化，这里顺序读取就好
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,  # 验证样本
        sampler=SequentialSampler(test_dataset),  # 顺序选取小批量
        batch_size=args.batch_size,
        drop_last=True,
    )

    # 验证集不需要随机化，这里顺序读取就好
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,  # 验证样本
        sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
        batch_size=args.batch_size,
        drop_last=True
    )
    # 用于存储加载的张量的列表
    all_tensors = []
    # 逐个加载文件中的张量并添加到列表中
    for file_path in range(args.n_labels):
        loaded_tensor = torch.load('./def_vec/tcm_sd/label_feature{}.pt'.format(file_path))
        all_tensors.append(loaded_tensor)
    label_feature = torch.stack(all_tensors, dim=0).to(args.device).view(-1, args.define_length, 1024)

    return train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict, label_feature