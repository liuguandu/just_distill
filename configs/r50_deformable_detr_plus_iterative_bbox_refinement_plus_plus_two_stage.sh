#!/usr/bin/env bash

set -x

# EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
EXP_DIR=exps/voc2007_lat10_incre_finetune3_allin
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    ${PY_ARGS}
