BASE_DIR=../results/linear_probing
FILES=(
    CLIP_FT_seed2795.pickle
    CLIP_FT_seed2859.pickle
    CLIP_FT_seed3231.pickle
    CLIP_seed1542.pickle
    CLIP_seed2859.pickle
    CLIP_seed3231.pickle
)
OUTPUT_FILE=../results/linear_probing/lp_results.csv

python3 convert_linear_probing_results.py --base_dir ${BASE_DIR} --files "${FILES[@]}" --out ${OUTPUT_FILE}