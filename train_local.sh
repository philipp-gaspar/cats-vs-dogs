echo "=========================="
echo "   Train model Locally.   "
echo "=========================="
echo " "


# BUCKET="qwiklabs-bucket-01"
# PROJECT_ID=$(gcloud config get-value project)
# REGION=us-central1
# JOB_NAME=cats_vs_dogs_$(date +%Y%m%d_%H%M%S)
#
# echo " - BUCKET: $BUCKET"
# echo " - PROJECT_ID: $PROJECT_ID"
# echo " - REGION: $REGION"
# echo " - JOB_NAME: $JOB_NAME"
# echo " "

gcloud ai-platform local train --package-path=${PWD}/trainer \
  --module-name=trainer.task \
  -- \
  --bucket $BUCKET
