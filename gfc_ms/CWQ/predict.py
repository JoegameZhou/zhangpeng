import os
import mindspore
from tqdm import tqdm
from collections import defaultdict
from IPython import embed


def validate(model, data, p, thresholds=0.5, verbose=False):
    count = 0
    correct = 0
    num_answers_pred_total = 0  # TP + FP
    TP_total = 0
    total_ans = 0
    for batch in tqdm(data, total=len(data)):

        topic_entity, question, answers,triples, entity_range = batch[0], batch[1], batch[2], batch[3], batch[4]
        outputs = model.construct( topic_entity, question, answers, triples, entity_range, training=False) # [bsz, Esize]
        hops = 1
        labels = mindspore.ops.nonzero(answers)
        answer_list = []
        for x in labels:
            answer_list.append(x[0])
        e_score = outputs['e_score']
        # e_score_answers = mindspore.numpy.where(e_score >= thresholds, )
        e_score_answers = mindspore.ops.nonzero(e_score>thresholds)
        num_pred = len(e_score_answers)
        num_answers_pred_total += num_pred
        total = num_answers_pred_total
        total_ans += len(answer_list)
        TP = 0
        for i in range(len(e_score_answers)):
            if e_score_answers[i][0] in answer_list:
                TP += 1
        TP_total += TP
        # topic_entities_idx = mindspore.ops.nonzero(topic_entities)
        # for item in topic_entities_idx:
        #     e_score[item[0], item[1]] = 0
        idx, scores = mindspore.ops.ArgMaxWithValue(axis = -1)(e_score) # [bsz], [bsz]
        # match_score = mindspore.ops.GatherD(labels, 1, idx.unsqueeze(-1)).squeeze().tolist()
        if verbose:
            answers = batch[2]
            if scores == 0:
                print('================================================================')
                question_ids = batch[1]['input_ids'].tolist()
                question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                print(' '.join(question_tokens))
                topic_id = batch[0].argmax(0).item()
                print('> topic entity: {}'.format(data.id2ent[topic_id]))
                for t in range(2):
                    print('>>>>>>> step {}'.format(t))
                    tmp = ' '.join(['{}: {:.3f}'.format(x, y) for x, y in
                                    zip(question_tokens, outputs['word_attns'][t].tolist())])
                    print('> Attention: ' + tmp)
                    print('> Relation:')
                    rel_idx = outputs['rel_probs'][t].gt(0.9).nonzero().squeeze(1).tolist()
                    for x in rel_idx:
                        print('  {}: {:.3f}'.format(data.id2rel[x], outputs['rel_probs'][t][x].item()))

                    print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in
                                                           outputs['ent_probs'][t].gt(0.8).nonzero().squeeze(
                                                               1).tolist()])))
                print('----')
                print('> max is {}'.format(data.id2ent[idx.item()]))
                print('> golden: {}'.format(
                    '; '.join([data.id2ent[_] for _ in answers.gt(0.9).nonzero().squeeze(1).tolist()])))
                print('> prediction: {}'.format(
                    '; '.join([data.id2ent[_] for _ in e_score.gt(0.9).nonzero().squeeze(1).tolist()])))
                print(' '.join(question_tokens))
                print(outputs['hop_attn'].tolist())
    # precision = TP_total / (num_answers_pred_total + 0.1)
    precision = TP_total / (total_ans + 0.1)
    p_info = ("precision: {}".format(precision))
    return p_info

