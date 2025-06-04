for seed in 1 2 3 4 5
do
python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset mimic --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_forecast --n_te_gptlayer 3 \
    --gpu 0 --n_st_gptlayer 3  --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
    --history 24 --task forecasting

python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset mimic --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_vector_forecast --collate vector --n_te_gptlayer 3 \
    --gpu 0 --n_st_gptlayer 3  --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
    --history 24 --task forecasting

python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset mimic --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_set_forecast --collate vector --n_te_gptlayer 3 \
    --gpu 0 --n_st_gptlayer 3  --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
    --history 24 --task forecasting
done