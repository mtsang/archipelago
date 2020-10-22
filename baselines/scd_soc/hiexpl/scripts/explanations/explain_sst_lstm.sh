#!/bin/bash

model_path=${1}
lm_path=${2}
python explain.py --resume_snapshot ${1}  --task sst --method cd --batch_size 1 --exp_name .cd2019.3 --nb_range 10 --lm_path models/sst_lm_2/best_snapshot_devloss_11.634532430897588_iter_1300_model.pt --nb_method ngram --gpu 0 --sample_n 20 --start 0 --stop 100 --dataset test


python explain.py --resume_snapshot results/best_snapshot_devacc_84.05963134765625_devloss_0.42193958163261414_iter_3700_model.pt  --task sst --method soc --batch_size 1 --exp_name .cd2019.3 --nb_range 10 --lm_path model/best_snapshot_devloss_11.708949835404105_iter_2000_model.pt --nb_method ngram --gpu 0 --sample_n 20 --start 0 --stop 100 --dataset test


python explain.py --resume_snapshot results/best_snapshot_devacc_84.05963134765625_devloss_0.42193958163261414_iter_3700_model.pt  --task sst --method soc --batch_size 1 --exp_name test2020.3 --nb_range 10 --lm_path model/best_snapshot_devloss_11.708949835404105_iter_2000_model.pt --nb_method ngram --gpu 0 --sample_n 20 --start 0 --stop 100 --dataset test

python explain.py --explain_model bert --resume_snapshot models_sst  --task sst --method soc --batch_size 1 --exp_name bertsoc2020.3 --nb_range 10 --lm_path model/best_snapshot_devloss_11.708949835404105_iter_2000_model.pt --nb_method ngram --gpu 0 --sample_n 20 --start 0 --stop 100 --dataset test --use_bert_tokenizer

python explain.py --explain_model bert --resume_snapshot models_sst  --task sst --method scd --batch_size 1 --exp_name bert2020.3 --nb_range 10 --lm_path model/best_snapshot_devloss_11.708949835404105_iter_2000_model.pt --nb_method ngram --gpu 0 --sample_n 20 --start 0 --stop 100 --dataset test --use_bert_tokenizer
