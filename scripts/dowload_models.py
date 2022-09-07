import click
import wget
import os
from rich import print
import zipfile


def get_download_link(model_name):
    return f"https://www.vascularmodel.com/svprojects/{model_name}.zip"


@click.command()
@click.argument("target_folder")
@click.argument("models", nargs=-1)
def main(target_folder, models):

    for model in models:
        print("Dowloading", model)
        source = get_download_link(model)
        target = os.path.join(target_folder, model)
        # wget.download(source, target + ".zip")
        print("Unzipping", model)
        with zipfile.ZipFile(target + ".zip", "r") as zip_ref:
            zip_ref.extractall(target_folder)
        print("Cleaning up", model)
        os.remove(target + ".zip")


if __name__ == "__main__":
    main()
