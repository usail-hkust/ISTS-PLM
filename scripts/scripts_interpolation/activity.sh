for seed in 1 2 3 4 5
do
python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset activity --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_forecast --n_te_gptlayer 6 \
    --gpu 0 --n_st_gptlayer 6  --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
    --history 3000 --task imputation

python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset activity --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_vector_forecast --collate vector --n_te_gptlayer 6 \
    --gpu 0 --n_st_gptlayer 6  --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
    --history 3000 --task imputation

python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset activity --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_set_forecast --collate vector --n_te_gptlayer 6 \
    --gpu 0 --n_st_gptlayer 6  --dropout 0.1 \
    --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
    --history 3000 --task imputation
done

    