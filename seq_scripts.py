import os
import csv
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

# [MLE FIX] Import AMP for memory optimization
from torch.cuda.amp import autocast, GradScaler

# [MLE FIX] Use the working evaluation import (removes the broken python_evaluation import)
from evaluation.slr_eval.wer_calculation import evaluate

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    # [MLE FIX] Initialize Scaler
    scaler = GradScaler()

    for batch_idx, data in enumerate(tqdm(loader)):
        data = device.dict_data_to_device(data)
        
        # [MLE FIX] Automatic Mixed Precision (AMP)
        with autocast():
            ret_dict = model(data)
            loss, loss_details = model.get_loss(ret_dict, data)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data['origin_info'])
            continue
        
        optimizer.zero_grad()
        
        # [MLE FIX] Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                f'\tEpoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. Loss: {loss.item():.2f}  lr:{clr[0]:.6f}'
            )
            recoder.print_log(
                "\t"
                + ", ".join([f"{k}: {v.item():.2f}" for k, v in loss_details.items()])
            )
        
        # [MLE FIX] aggressive garbage collection
        del ret_dict
        del loss
        del loss_details
        del data
    
    optimizer.scheduler.step()
    # [MLE FIX] Clear cache
    torch.cuda.empty_cache()
    
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value

def seq_eval(
    cfg, loader, model, device, mode, epoch, work_dir, recoder, task, evaluate_tool="python"
):
    model.eval()
    total_info = []
    total_sent_fusion = []
    total_sent_conv_fusion = []
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        data = device.dict_data_to_device(data)
        with torch.no_grad():
            ret_dict = model(data)

        total_info += [file_name.split("|")[0] for file_name in data['origin_info']]
        total_sent_fusion += ret_dict['recognized_sents_fusion']
        total_sent_conv_fusion += ret_dict['conv_sents_fusion']
        
        # [MLE FIX] Free memory
        del data
        del ret_dict

    python_eval = True if evaluate_tool == "python" else False
    write2file(
        work_dir + "output-hypothesis-fusion-{}.ctm".format(mode), total_info, total_sent_fusion
    )
    write2file(
        work_dir + "output-hypothesis-conv-fusion-{}.ctm".format(mode), total_info, total_sent_conv_fusion
    )
    
    # [MLE FIX] Clear cache
    torch.cuda.empty_cache()

    if mode == 'test':
        csv_file = f'{work_dir}test.csv'
        if task == 'us':
            ctm_file = f'{work_dir}output-hypothesis-conv-fusion-test.ctm'
        elif task == 'si':
            ctm_file = f'{work_dir}output-hypothesis-fusion-test.ctm'
        with open(ctm_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        data = {}
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 5:  
                id = parts[0]  
                word = parts[4]  
                if id not in data:
                    data[id] = []
                data[id].append(word)

        data = dict(sorted(data.items(), key=lambda item: int(item[0])))

        with open(csv_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "gloss"])
            for id, words in data.items():
                gloss = " ".join(words)
                writer.writerow([id, gloss])
        return csv_file
    else:
        try:
            lstm_ret_fusion = evaluate(
                prefix=work_dir,
                mode=mode,
                output_file="output-hypothesis-fusion-{}.ctm".format(mode),
                evaluate_dir=cfg.dataset_info['evaluation_dir'],
                evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
                triplet=True,
            )
            conv_ret_fusion = evaluate(
                prefix=work_dir,
                mode=mode,
                output_file="output-hypothesis-conv-fusion-{}.ctm".format(mode),
                evaluate_dir=cfg.dataset_info['evaluation_dir'],
                evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
                # triplet=True,
            )
        except:
            print("Unexpected error:", sys.exc_info()[0])
            lstm_ret_fusion = 100.0
            conv_ret_fusion = 100.0
        finally:
            pass
        recoder.print_log(
            f"Epoch {epoch}, {mode} Conv1D WER: {conv_ret_fusion: 2.2f}%, BiLSTM WER: {lstm_ret_fusion: 2.2f}%", f"{work_dir}/{mode}.txt"
        )
        return min([conv_ret_fusion, lstm_ret_fusion])       

def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(
                    info[sample_idx],
                    word_idx * 1.0 / 100,
                    (word_idx + 1) * 1.0 / 100,
                    word[0],
                )
            )