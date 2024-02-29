# Args to the function call
LOGFILE=/projects/mluq/model-store/extrap10d/ExtrapLRM10D.log
FIGDIR=/projects/mluq/model-store/extrap10d/Figures/
CKPTDIR=/projects/mluq/model-store/extrap10d/lrm/ckpts/
REFINPUT=/home/gpash/S-mluq-prop/mluqprop/BNN/inputs/bestLRM10D.in
INPUT=/home/gpash/S-mluq-prop/mluqprop/BNN/inputs/extrap.in
NPREDICT=400

#python postproc_extrap.py \
# --logfile $LOGFILE \
# --figdir $FIGDIR \
# --ckptdir $CKPTDIR \
# --refinput $REFINPUT \
# --input $INPUT \
# --npredict $NPREDICT \
# --no-short 

python postproc_10d_extrap.py \
 --figdir $FIGDIR \
 --ckptdir $CKPTDIR \
 --refinput $REFINPUT \
 --input $INPUT \
 --npredict $NPREDICT \
