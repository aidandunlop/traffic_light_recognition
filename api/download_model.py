import boto3
import os

temp_modal_path = "/tmp/tlr_model.pth"


def download_model(save_location):
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        file = s3.download_file(os.environ["AWS_BUCKET"], "model.pth", save_location)
    except boto3.exceptions.ResourceNotExistsError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise

    return file


if __name__ == "__main__":
    print("Downloading model from S3...")
    download_model(temp_modal_path)
    print("Model downloaded.")
