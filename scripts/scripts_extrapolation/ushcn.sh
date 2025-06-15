
gpu=0

for seed in 1 2 3 4 5
do

python regression.py \
    --batch 16 --lr 1e-3 --state 'def' --epoch 1000 --patience 10 \
    --dataset ushcn --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_forecast --dropout 0 \
    --gpu $gpu --n_te_plmlayer 1 --n_st_plmlayer 1 \
    --te_model bert --st_model bert --sample_rate 1 --semi_freeze \
    --history 24 --task forecasting

done

    