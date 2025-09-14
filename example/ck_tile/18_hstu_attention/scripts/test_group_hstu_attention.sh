#!/bin/bash

BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

ndist=0

if [ $# -ge 2 ]; then
    ndist=$1
fi

for dtype in "fp16" "bf16"; do
    set -x

    ## no masking
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=0 -targets=0 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=0,3,0 -g_context_lens=0,0,0 -g_minfull_lens=0,0,0 -g_attn_scales=0,0.1,0

    ## causal
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=1 -targets=0 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=0,3,0 -g_context_lens=0,0,0 -g_minfull_lens=0,0,0 -g_attn_scales=0,0.1,0

    ## causal+local
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=1 -targets=0 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=5,5,5 -g_context_lens=0,0,0 -g_minfull_lens=0,0,0 -g_attn_scales=0,0.1,0

    ## causal+local+context
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=1 -targets=0 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=5,5,5 -g_context_lens=8,8,8 -g_minfull_lens=7,7,7 -g_attn_scales=0,0.1,0

    ## causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=1 -targets=8 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=5,5,5 -g_context_lens=8,8,8 -g_minfull_lens=7,7,7 -g_attn_scales=0,0.1,0

    ##  no-causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=0 -targets=8 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=5,5,5 -g_context_lens=8,8,8 -g_minfull_lens=7,7,7 -g_attn_scales=0,0.1,0

    ## causal+local+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=1 -targets=8 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=5,5,5 -g_context_lens=8,8,8 -g_minfull_lens=290,290,290 -g_attn_scales=0,0.1,0

    ## causal+local+context+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=1 -targets=8 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=5,5,5 -g_context_lens=8,8,8 -g_minfull_lens=290,290,290 -g_attn_scales=0,0.1,0

    ## no-causal+local+context+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=18 -g=3 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310,308,312 -causal=0 -targets=8 -norm_dist=$ndist \
      -g_max_seqlens=310,312,312 -g_local_lens=5,5,5 -g_context_lens=3,3,3 -g_minfull_lens=290,290,290 -g_attn_scales=0,0.1,0

    set +x
done

set -x
## special cases
$EXE -v=1 -prec="bf16" -b=16 -g=2 -nhead=109 -hdim_qk=16 -hdim_v=16 -causal=1 \
        -seqlens=89,84,80,60,69,78,67,61,65,98,94,85,88,60,89,84 \
	-targets=20,4,7,5,3,16,7,5,15,11,6,16,14,11,15,11 \
	-norm_dist=$ndist -alpha=0.25 -g_max_seqlens=120,120 -g_local_lens=13,2 -g_context_lens=0,0 -g_minfull_lens=0,0 -g_attn_scales=0.0083,0.0083

$EXE -v=1 -prec="bf16" -b=8 -g=2 -nhead=4 -hdim_qk=16 -hdim_v=16 -causal=1 \
        -seqlens=81,77,91,72,95,87,73,88 -targets=5,11,4,15,1,18,4,8 \
        -norm_dist=$ndist -alpha=0.25 -g_max_seqlens=120,120 -g_local_lens=13,2 -g_context_lens=0,0 -g_minfull_lens=0,0 -g_attn_scales=0.0083,0.0083

set +x
