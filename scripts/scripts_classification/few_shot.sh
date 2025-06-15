gpu=0
for split in 1 2 3 4 5
do

python classification.py --few_shot\
    --batch 6 --lr 1e-3 --epoch 20 --patience 20\
    --task 'P12' --seed 0 --d_model 768 --max_len -1\
    --n_classes 2 --gpu $gpu\
    --model istsplm --state 'def' --n_te_plmlayer 6 \
    --n_st_plmlayer 6 --semi_freeze --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --split $split

python classification.py --few_shot\
    --batch 6 --lr 1e-3 --epoch 20 --patience 20 \
    --task 'P19' --seed 0 --d_model 768 --max_len -1\
    --n_classes 2 --gpu $gpu\
    --model istsplm --collate vector --state 'def' --n_te_plmlayer 6 \
    --n_st_plmlayer 6 --semi_freeze --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --split $split

python classification.py --few_shot\
    --batch 6 --lr 1e-3 --epoch 20 --patience 20 \
    --task 'PAM' --seed 0 --d_model 768 --max_len -1\
    --n_classes 8 --gpu $gpu\
    --model istsplm --collate vector --state 'def' --n_te_plmlayer 6 \
    --n_st_plmlayer 6 --semi_freeze --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --split $split

done