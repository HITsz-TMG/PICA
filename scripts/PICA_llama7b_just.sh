conda activate PICA
cd /GLOBALFS/hitsz_bthu_1/lzy/brainstrom/prior_token_generation/github

export OUTPUT="XXX"

python PICA.py \
--data_path /GLOBALFS/hitsz_bthu_1/lzy/brainstrom/data/just_eval.json \
--model_path /GLOBALFS/hitsz_bthu_1/lzy/Model/llama-2-7b \
--device 0 \
--log \
--use_cache \
--output_dir $OUTPUT \
--max_new_tokens 4096 \
--model_max_length 4096 \
--eos_token_id "13,28956,13" \
--prior_token_num 10 \
--interventation_layer 9 \
--icl_mode PICA \
--config CFG3 \
--dummy just
