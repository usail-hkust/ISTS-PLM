
gpu=0

for seed in 1 2 3 4 5
do

python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset activity --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_forecast --n_te_plmlayer 6 \
    --gpu $gpu --n_st_plmlayer 6  --dropout 0.1 \
    --te_model bert --st_model bert --sample_rate 1 --semi_freeze\
    --history 3000 --task forecasting

# python regression.py \
#     --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
#     --dataset activity --seed $seed --d_model 768 --max_len -1 \
#     --model istsplm_vector_forecast --n_te_plmlayer 6 \
#     --gpu $gpu --n_st_plmlayer 6  --collate vector --dropout 0.1 \
#     --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
#     --history 3000 --task forecasting

# python regression.py \
#     --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
#     --dataset activity --seed $seed --d_model 768 --max_len -1 \
#     --model istsplm_set_forecast --collate vector --n_te_plmlayer 6 \
#     --gpu $gpu --n_st_plmlayer 6  --dropout 0.1 \
#     --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
#     --history 3000 --task forecasting
done

    