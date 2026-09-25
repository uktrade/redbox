This file is for recording set up process while working on the dev happiness ticket, this should be integrated into the readme before release

# Local S3 setup
Redbox uses s3 buckets for storage, even in local environments. To set this up, you will need to create your own bucket in the redbox AWS **development account**, the aws cli command for which you can find below:
```
aws s3api create-bucket \
  --profile <your-redbox-dev-aws-profile> \
  --bucket redbox-developer-bucket-<your-name> \
  --region eu-west-2 \
  --create-bucket-configuration LocationConstraint=eu-west-2
```
Then you will need to set BUCKET_NAME in the .env file to your bucket name. To test, restart your docker containers and then try to upload a file through the frontend.
