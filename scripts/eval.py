import argparse
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
import pdb
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generative model results")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to inference result file")
    parser.add_argument("--task", type=str, required=True,
                        help="Task to evaluate")
    parser.add_argument("--label_file", type=str, default=None,
                        help="Path to label file (for ranking tasks)")
    return parser.parse_args()

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    args = parse_args()
    
    dataset = load_jsonl(args.result_file)
    label_list = []
    prediction_list = []
    for data in dataset:
        if data["task"] == args.task:
            label_list.append(data["answer"])
            prediction_list.append(data["output"])
    
    if not label_list:
        print(f"No data found for task: {args.task}")
        return

    #evaluation for extraction tasks
    if (args.task == 'Attribute_Value_Extraction'):

        #load predictions as dictionary
        idx2pred = {}
        pred_skipped = 0
        avg_pred = 0
        for i in range(len(prediction_list)):
            try:
                idx2pred[i] = json.loads(prediction_list[i])
                if type(idx2pred[i]) != list:
                    idx2pred[i] = []
                else:
                    avg_pred += len(idx2pred[i])
            except:
                idx2pred[i] = []
                pred_skipped += 1
                continue

        avg_pred /= len(idx2pred)
        attribute2count_label = {}
        attribute2count_pred = {}
        attribute2hit = {}
        label_skipped = 0

        #count the total number of hits
        for i in range(len(label_list)):

            try:
                labels = json.loads(label_list[i])
            except:
                label_skipped += 1
                continue

            for label in labels:

                if label["attribute"] not in attribute2count_label:
                    attribute2count_label[label["attribute"]] = 0
                attribute2count_label[label["attribute"]] += 1

                for pred in idx2pred[i]:
                    
                    if "attribute" in pred:
                        if pred["attribute"] not in attribute2count_pred:
                            attribute2count_pred[pred["attribute"]] = 0
                        attribute2count_pred[pred["attribute"]] += 1
                    
                    if ("attribute" in pred and "value" in pred and "source" in pred):

                        if (type(pred) == dict) and (pred["attribute"] == label["attribute"] and pred["value"] == label["value"] and pred["source"] == label["source"]):
                            if pred["attribute"] not in attribute2hit:
                                attribute2hit[pred["attribute"]] = 0
                            attribute2hit[pred["attribute"]] += 1

                        elif type(pred) and (pred["attribute"] == label["attribute"]) and (pred["value"] != None) and (pred["source"] != None)\
                        and ((str(pred["value"]) in str(label["value"])) or (str(label["value"]) in str(pred["value"]))) and ((pred["source"] in label["source"]) or (label["source"] in pred["source"])):
                            if pred["attribute"] not in attribute2hit:
                                attribute2hit[pred["attribute"]] = 0
                            attribute2hit[pred["attribute"]] += 1

        total_hit = 0
        total_count_label = 0
        total_count_pred = 0
        for k in attribute2hit:
            total_hit += attribute2hit[k]
        
        for k in attribute2count_label:
            total_count_label += attribute2count_label[k]
        
        for k in attribute2count_pred:
            total_count_pred += attribute2count_pred[k]

        total_recall = total_hit / total_count_label
        total_precision = total_hit / (total_count_pred + 1e-6)
        total_f1 = (2 * total_recall * total_precision) / (total_recall + total_precision + 1e-6) 
        print('{"recall": %.3f, "precision": %.3f, "f1": %.3f, "#invalid": %d}' % (total_recall, total_precision, total_f1, pred_skipped))
    
    #evaluation for binary classification tasks
    elif (args.task == 'Answerability_Prediction') \
        or (args.task == 'Product_Substitue_Identification') \
        or (args.task == 'Product_Matching'):
        skipped = 0
        filtered_prediction_list = []
        filtered_label_list = []
        valid_response = ['yes', 'no']
        valid_response = set(valid_response)

        for i in range(len(prediction_list)):
            pred = prediction_list[i].lower().strip()
            if pred in valid_response:
                filtered_prediction_list.append(pred)
                filtered_label_list.append(label_list[i])
            else:
                if label_list[i]=='yes':
                    filtered_prediction_list.append('no')
                else:
                    filtered_prediction_list.append('yes')
                filtered_label_list.append(label_list[i])
                skipped += 1

        acc = accuracy_score(filtered_label_list, filtered_prediction_list)
        pre_yes = precision_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes', average = 'binary')
        rec_yes = recall_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes', average = 'binary')
        f1_yes  = f1_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes', average = 'binary')

        tn, fp, fn, tp = confusion_matrix(filtered_label_list, filtered_prediction_list).ravel()
        specificity = tn / (tn + fp + 1e-8)
        NPV = tn / (tn + fn + 1e-8)

        print('{"acc": %.3f, "recall": %.3f, "precision": %.3f, "f1": %.3f, "specificity": %.3f, "npr": %.3f, "#invalid": %d}' % (acc, rec_yes, pre_yes, f1_yes, specificity, NPV, skipped))
    
    #evaluation for recommendation tasks
    elif (args.task == 'Sequential_Recommendation'):
        skipped = 0
        filtered_prediction_list = []
        filtered_label_list = []
        valid_response = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
        valid_response = set(valid_response)

        for i in range(len(prediction_list)):
            pred = prediction_list[i].strip()
            if pred in valid_response:
                filtered_prediction_list.append(pred)
                filtered_label_list.append(label_list[i])
            else:
                if label_list[i]=='A':
                    filtered_prediction_list.append('B')
                else:
                    filtered_prediction_list.append('A')
                filtered_label_list.append(label_list[i])
                skipped += 1
        
        acc = accuracy_score(filtered_label_list, filtered_prediction_list)
        print('{"recall@1": %.3f, "#invalid": %d}' % (acc, skipped))
    
    #evaluation for multi-class classification tasks
    elif (args.task == 'Multiclass_Product_Classification') \
        or (args.task == 'Sentiment_Analysis') \
        or (args.task == 'Product_Relation_Prediction'):

        skipped = 0
        filtered_prediction_list = []
        filtered_label_list = []
        valid_response = ['A', 'B', 'C', 'D', 'E']
        valid_response = set(valid_response)

        for i in range(len(prediction_list)):
            pred = prediction_list[i].strip()
            if pred[0] in valid_response:
                filtered_prediction_list.append(pred[0])
                filtered_label_list.append(label_list[i][0])
            else:
                if label_list[i][0] == 'A':
                    filtered_prediction_list.append('B')
                else:
                    filtered_prediction_list.append('A')
                filtered_label_list.append(label_list[i][0])
                skipped += 1

        acc = accuracy_score(filtered_label_list, filtered_prediction_list)
        pre_macro = precision_score(filtered_label_list, filtered_prediction_list, average = 'macro')
        rec_macro = recall_score(filtered_label_list, filtered_prediction_list, average = 'macro')
        f1_macro  = f1_score(filtered_label_list, filtered_prediction_list, average = 'macro')

        print('{"acc": %.3f, "recall": %.3f, "precision": %.3f, "f1": %.3f, "#invalid": %d}' % (acc, rec_macro, pre_macro, f1_macro, skipped))
    
    #evaluation for generation tasks
    elif (args.task == 'Answer_Generation' or args.task == 'Query_Rewriting' or args.task == 'item_profile' or args.task == 'user_profile'):
    
        from bert_score import score
        import torch
        import numpy as np
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # # To be modified
        # threshold = 2000
        # for i in range(len(prediction_list)):
        #     if len(prediction_list[i].split()) > threshold:
        #         pred = prediction_list[i].split()
        #         pred = pred[:threshold]
        #         prediction_list[i] = ' '.join(pred)
        
        # for i in range(len(label_list)):
        #     if len(label_list[i].split()) > threshold:
        #         label = label_list[i].split()
        #         label = label[:threshold]
        #         label_list[i] = ' '.join(label)

        num_skipped = 0
        bert_score = {}
        (P,R,F), hashname = score(prediction_list, label_list, lang="en", return_hash=True, model_type='/inspire/hdd/global_user/zhangweinan-24046/longformer-large-4096', num_layers=14, batch_size=8)
        bert_score['precision'] = P.mean()
        bert_score['recall'] = R.mean()
        bert_score['f1'] = F.mean()
        print(f"{hashname}: P={bert_score['precision']:.6f} R={bert_score['recall']:.6f} F={bert_score['f1']:.6f}")
        torch.cuda.empty_cache()

        # from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
        # path = '/data/zhelizhou/gritlm/evaluation/bleurt-20-D12'
        # config = BleurtConfig.from_pretrained(path)
        # model = BleurtForSequenceClassification.from_pretrained(path).to(device)
        # tokenizer = BleurtTokenizer.from_pretrained(path)

        # model.eval()
        # res = []

        # with torch.no_grad():
        #     for i in range(len(label_list)):
        #         inputs = tokenizer(label_list[i], prediction_list[i], return_tensors='pt', padding='longest').to(device)
        #         try:
        #             tmp = model(**inputs).logits.flatten()
        #         except:
        #             tmp = ''
        #             num_skipped += 1
        #             continue
        #         tmp = model(**inputs).logits.flatten()
        #         res.append(tmp[0].cpu())
        # bleurt_score = np.mean(res)
        print('{"recall": %.3f, "precision": %.3f, "f1": %.3f, "#invalid": %d}' % (bert_score['recall'], bert_score['precision'], bert_score['f1'], num_skipped))
    
    #evaluation for ranking tasks
    elif (args.task == 'Query_Product_Ranking'):
        import numpy as np

        def DCG(score_list):

            dcg = 0
            for i in range(len(score_list)):
                dcg += (2 ** score_list[i] - 1) / (np.log2(i + 2))
            return dcg
        
        score_mapping = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0}
        label2score = {}

        option_labels_list = json.load(open('ECInstruct/Query_Product_Ranking/IND_Diverse_Instruction/label.json', 'r'))
        counter = 0
        for option_labels in option_labels_list:
            label2score[counter] = {}
            option = 'A'
            for option_label in option_labels:
                label2score[counter][option] = option_label
                option = chr(ord(option) + 1)
            counter += 1

        total_ndcg = 0
        skipped = 0
        for i in range(len(prediction_list)):
            scores = []
            ranks = prediction_list[i].strip().split(',')
            for rank in ranks:
                try:
                    scores.append(score_mapping[label2score[i][rank[0]]])
                except:
                    skipped += 1
                    continue
            
            ideal_scores = sorted(scores, reverse=True)

            dcg  = DCG(scores)
            idcg = DCG(ideal_scores)
            total_ndcg += (dcg / (idcg + 1e-6))
        
        avg_ndcg = total_ndcg / len(label_list)
        print('{"NDCG": %.3f, "#invalid": %d}' % (avg_ndcg, skipped))



if __name__ == '__main__':
    main()