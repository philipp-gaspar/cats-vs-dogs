echo "=============================="
echo "   Train model on Cloud ML.   "
echo "=============================="
echo " "

# if [ "$#" -ne 1 ]; then
#     echo "Usage:    ./train_cloudml.sh destination-bucket-name"
#     echo "   eg:    .train_cloudml.sh qwiklabs-bucket-01"
#     exit -1
# fi

BUCKET="qwiklabs-bucket-01"
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
JOB_NAME=cats_vs_dogs_$(date +%Y%m%d_%H%M%S)

echo " - BUCKET: $BUCKET"
echo " - PROJECT_ID: $PROJECT_ID"
echo " - REGION: $REGION"
echo " - JOB_NAME: $JOB_NAME"
echo " "

gcloud ai-platform jobs submit training $JOB_NAME \
  --staging-bucket=gs://$BUCKET \
  --region=$REGION \
  --module-name=trainer.task \
  --python-version=3.5 \
  --runtime-version=1.14 \
  --package-path=${PWD}/trainer \
  --scale-tier=STANDARD_1 \
  -- \
  --bucket $BUCKET
