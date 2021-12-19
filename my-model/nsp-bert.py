#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/9/11

import time
import numpy
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from utils import *
import os
import json
import json_lines
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, pattern="", is_pre=True, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.is_pre = is_pre

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, text in self.sample(random):
            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=self.pattern, second_text=text, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text, second_text=self.pattern, maxlen=maxlen)
            source_ids, target_ids = token_ids[:], token_ids[:]
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids, = [], []

def predict(data_generator_list, data):
    print("\n*******************Start to Zero-Shot predict*******************", flush=True)
    patterns_logits = [[] for _ in answers]
    samples_logits = [[] for _ in data]
    for i in range(len(data_generator_list)):
        data_generator = data_generator_list[i]
        counter = 0
        for (x, _) in data_generator:
            outputs = model.predict(x[:2])
            for out in outputs:
                logit_pos = out[0].T
                patterns_logits[i].append(logit_pos)
                samples_logits[counter].append(logit_pos)
                counter += 1
    preds = []
    for i in range(len(patterns_logits[0])):
        pred = numpy.argmax([logits[i] for logits in patterns_logits])
        preds.append(int(pred))
    return preds, samples_logits

if __name__ == "__main__":

    # Load the hyper-parameters-----------------------------------------------------------
    maxlen = 128  # The max length 128 is used in our paper
    batch_size = 40  # Will not influence the results

    # Choose a model----------------------------------------------------------------------
    # Recommend to use 'uer-mixed-bert-base'
    # model_names = ['google-bert', 'google-bert-small', 'google-bert-zh',
    #                'hfl-bert-wwm', 'hfl-bert-wwm-ext',
    #                'uer-mixed-bert-tiny', 'uer-mixed-bert-small',
    #                'uer-mixed-bert-base', 'uer-mixed-bert-large']
    model_name = 'uer-mixed-bert-base'

    # Load model and dataset class
    bert_model = Model(model_name=model_name)
    tokenizer = Tokenizer('.' + bert_model.dict_path, do_lower_case=True)
    # Load BERET model with NSP head
    model = build_transformer_model(
        config_path='.' + bert_model.config_path, checkpoint_path='.' + bert_model.checkpoint_path, with_nsp=True,
    )

    mapping = {0:'A',
               1:'B',
               2:'C',
               3:'D',
               4:'E'}



    with open('../dev.jsonl', 'r') as f:
        data = [item for item in json_lines.reader(f)]
    num = 500
    start = time.time()
    count = 0
    for i in tqdm(range(num)):
        ground_truth = data[i]['answerKey']
        question_answer = data[i]['question']
        answers = [item['text'] for item in question_answer['choices']]
        # answer_prompt = ['the answer is {}'.format(i) for i in answers]
        is_pre = False
        question = [data[i]['question']['stem']]
        generate_list = []
        for each_answer in answers:
            generate_list.append(data_generator(pattern=each_answer, is_pre=is_pre, data=question, batch_size=batch_size))

        preds, samples_logits = predict(generate_list, question)
        pred_label = answers[preds[0]]
        print(preds[0])
        if mapping[preds[0]] == ground_truth:
            count+=1
            print('true')
        print("Sample {}:".format(i))
        print("Original Text: {}".format(question))
        print("Predict label: {}".format(pred_label))
        print("Logits: {}".format(samples_logits[0]))
        print(ground_truth)
    print('正确率为',count/num)
    end = time.time()
    print('花费时长为',end-start,'s')
