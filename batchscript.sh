#!/bin/bash
#SBATCH --job-name=nlp_first_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=12:00:00
#SBATCH -p gtx

# Load any required modules or activate a conda environment if needed

# Run your job commands here
# python3 run.py --do_eval --task nli --dataset snli.json --model ./both_hybrid_model/ --output_dir ./eval_hybrid_both_filter_contradiction/ --hybrid True --biased_model_type both
# python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/
# python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model_hypothesis_only/ --hypothesis_only True
python3 run.py --do_train --task nli --dataset snli --output_dir ./hybrid_hypothesis_only_one_epoch/ --hybrid True --biased_model_type hypothesis_only
# python3 run.py --do_train --task nli --dataset snli --output_dir ./both_hybrid_model/ --hybrid True --biased_model_type both
# python3 run.py --do_train --task nli --dataset snli --output_dir ./both_hybrid_model/ --hybrid True --biased_model_type both