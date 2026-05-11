MODEL_PATH="/tmp/hitnet/models"
MODEL_NAME="eth3d.pb"
mkdir -p $MODEL_PATH
wget -P $MODEL_PATH -N https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/$MODEL_NAME
