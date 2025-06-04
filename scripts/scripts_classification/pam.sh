gpu=0
for split in 1 2 3 4 5
do

python classification.py \
    --batch 6 --lr 1e-3 --epoch 20 --patience 20\
    --task 'PAM' --seed 0 --d_model 768 --max_len -1\
    --n_classes 8 --gpu $gpu\
    --model istsplm --state 'def' --n_te_gptlayer 6 \
    --n_st_gptlayer 6 --semi_freeze --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --split $split

python classification.py \
    --batch 6 --lr 1e-3 --epoch 20 --patience 20 \
    --task 'PAM' --seed 0 --d_model 768 --max_len -1\
    --n_classes 8 --gpu $gpu\
    --model istsplm_vector --collate vector --state 'def' --n_te_gptlayer 6 \
    --n_st_gptlayer 6 --semi_freeze --dropout 0.1
    --te_model gpt --st_model bert --sample_rate 1 --split $split

python classification.py \
    --batch 6 --lr 1e-3 --epoch 20 --patience 20 \
    --task 'PAM' --seed 0 --d_model 768 --max_len -1\
    --n_classes 8 --gpu $gpu\
    --model istsplm_set --collate vector --state 'def' --n_te_gptlayer 6 \
    --n_st_gptlayer 6 --semi_freeze --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --split $split

done