# Run EEV
docker run --gpus all --ipc=host -it --rm \
        --user $UID:$GID \
        --volume="/etc/group:/etc/group:ro" \
        --volume="/etc/passwd:/etc/passwd:ro" \
        --volume="/etc/shadow:/etc/shadow:ro" \
        -v /home/hvthong/sXProject/EvokedExpression/dataset/eev2021:/mnt/sXProject/EvokedExpression \
        -v /mnt/XProject/EvokedExpression:/mnt/XProject/EvokedExpression \
        -w /mnt/XProject/EvokedExpression \
        eev:pytorch1.8.1 bash testing.sh

# Run MediaEval
#docker run --gpus all --ipc=host -it --rm \
#        --user $UID:$GID \
#        --volume="/etc/group:/etc/group:ro" \
#        --volume="/etc/passwd:/etc/passwd:ro" \
#        --volume="/etc/shadow:/etc/shadow:ro" \
#        -v /home/hvthong/sXProject/EvokedExpression/dataset:/mnt/sXProject/EvokedExpression/dataset \
#        -v /mnt/XProject/EvokedExpression:/mnt/XProject/EvokedExpression \
#        -w /mnt/XProject/EvokedExpression \
#        eev:pytorch1.8.1 bash scripts/mediaeval18_train.sh
