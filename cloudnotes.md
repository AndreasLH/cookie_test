gcloud compute instances create hello-mnist2 --zone="europe-north1-a" --image-family="pytorch-latest-cpu" --image-project=deeplearning-platform-release

docker pull gcr.io/optimal-brace-374308/testing:latest

gcloud beta compute ssh --zone europe-west4-a gpu-test --project optimal-brace-374308

python -c "import torch; print(torch.__version__)"

docker tag gcp_vm_tester gcr.io/optimal-brace-374308/gcp_vm_tester
docker push gcr.io/optimal-brace-374308/gcp_vm_tester

gcloud compute instances create-with-container instance-with-docker2 --container-image=gcr.io/optimal-brace-374308/gcp_vm_tester --zone europe-west1-b

python -c "import torch; print(torch.__version__)"
python -c "import matplotlib; print(matplotlib.__version__)"

publicize data

## vertex
gcloud ai custom-jobs create --region=europe-west1 --display-name=test-run --config=cgp_config.yaml
