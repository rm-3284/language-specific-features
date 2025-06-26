#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

run_intervention() {
       local intervention_lang=$1

       python "$DIR/../scripts/ppl.py" meta-llama/Llama-3.2-1B openlanguagedata/flores_plus \
              --split devtest \
              --layer "model.layers.{0..15}.mlp.act_fn" \
              --lang eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
              --start 0 --end 1000 \
              --intervention-type neuron \
              --intervention-lang "$intervention_lang" \
              --value 0.2 \
              --lape-result-path mlp_acts_specific/meta-llama/Llama-3.2-1B/lape_neuron.pt \
              --out-path "neuron_intervention/baseline/ppl_${lang}.pt" \
              --neuron-intervention-method scaling \
              --lape-value-type final_indice_global_max_active

       python "$DIR/../scripts/ppl.py" meta-llama/Llama-3.2-1B openlanguagedata/flores_plus \
              --split devtest \
              --layer "model.layers.{0..15}.mlp.act_fn" \
              --lang eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
              --start 0 --end 1000 \
              --intervention-type neuron \
              --intervention-lang "$intervention_lang" \
              --value -0.2 \
              --lape-result-path mlp_acts_specific/meta-llama/Llama-3.2-1B/lape_neuron.pt \
              --out-path "neuron_intervention/baseline/ppl_${lang}.pt" \
              --neuron-intervention-method scaling \
              --lape-value-type final_indice_global_max_active
}

langs=(
       "eng_Latn"
       "deu_Latn"
       "fra_Latn"
       "ita_Latn"
       "por_Latn"
       "hin_Deva"
       "spa_Latn"
       "tha_Thai"
       "bul_Cyrl"
       "rus_Cyrl"
       "tur_Latn"
       "vie_Latn"
       "jpn_Jpan"
       "kor_Hang"
       "cmn_Hans"
)

for lang in "${langs[@]}"; do
       run_intervention ${lang}
done
