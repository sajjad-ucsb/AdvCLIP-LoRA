
DATA="../DATA/"
BACKBONE=ViT-B/32
BACKBONE_NAME=ViT-B-32
ATTACK_TYPE=PGD
EPSILON=1.0
ROBUST=robust
EPSILON_TRAIN=1.0
TEST_TRAIN=train


for DATASET in imagenet
do
  for SHOT in 1 2 4 8 16
  do
    for SEED in 1
    do
      for M in 2
      do
        CHECKPOINTS=checkpoints-new/checkpoints-${ROBUST}-${M}/epsilon_train${EPSILON_TRAIN}
        DIR=results-new/${BACKBONE_NAME}/${ROBUST}-m${M}-attack${ATTACK_TYPE}-${TEST_TRAIN}/eps_train${EPSILON_TRAIN}
        if [ -d "$DIR/eps${EPSILON}-${DATASET}-shots${SHOT}-seed${SEED}.log" ]; then
            echo "Oops! The results exist at ${DIR/${DATASET}_shots${SHOT}_seed${SEED}.log} (so skip this job)"
        else
            echo "Running ${DATASET}, shot ${SHOT}, seed ${SEED}, M=${M}"
            mkdir -p "$DIR"

            nohup python main.py \
            --root_path ${DATA} \
            --save_path ${CHECKPOINTS} \
            --dataset ${DATASET} \
            --shots ${SHOT} \
            --seed ${SEED} \
            --m ${M} \
            --backbone ${BACKBONE} \
            --attack_type ${ATTACK_TYPE} \
            --epsilon ${EPSILON} \
            --epsilon_train ${EPSILON_TRAIN} \
            > ${DIR}/eps${EPSILON}-${DATASET}-shots${SHOT}-seed${SEED}.log 2>&1 &
            wait $!
        fi
      done
    done
  done
done

