import argparse
import requests
import os
import shutil


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# from this StackOverflow answer: https://stackoverflow.com/a/39225039
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


parser = argparse.ArgumentParser(description="Download data and pretrained models.")
parser.add_argument(
    "--quick_demo",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Downloads all data and pretrained models for the quick demo.",
)
parser.add_argument(
    "--demos",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Downloads all data and pretrained models for the demos.",
)
parser.add_argument(
    "--experiments",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Downloads all data and pretrained models for the experiments.",
)
parser.add_argument(
    "--all",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Downloads all data and pretrained models.",
)
parser.add_argument(
    "--downloads_folder",
    type=str,
    default="downloads",
    help="Name of downloads folder. All default code points to 'downloads'",
)


args = parser.parse_args()
quick_demo = args.quick_demo
demos = args.demos
experiments = args.experiments
download_all = args.all
downloads_folder = args.downloads_folder

all_options = [quick_demo, demos, experiments, download_all]

if not any(all_options):
    download_all = True

if sum(all_options) > 1:
    raise ValueError("Cannot enable multiple options.")


pretrained_model_ids = {
    "bert": "1sUqMqCqoZEjEuNEt6MQZQ2obJVm_r4Vt",
    "covid_net": "1aoZ9RTJeuAxPEMYo1ytYbmyQAJkaYIHo",
    "hiexpl_lm": "1hU9EmzdtL8s21PgnYxH6tSLfKSndggKf",
    "autoint": "1I2jzt_zLlmUB2RO3XjbHj1evmXN3CDuu",
}

data_ids = {
    "sst": "1iBrbVQrFzDfjWl-1Pv05tp-2IPeaGUFK",
    "avazu": "1NXzDvOFxAPMj4oAr_KgkEsIoURg0JCwH",
}

destination = "/home/myusername/work/myfile.ext"


if quick_demo:
    keys = ["bert", "sst"]

elif demos:
    keys = ["bert", "covid_net", "autoint", "sst", "avazu"]

elif experiments:
    keys = ["bert", "hiexpl_lm", "sst"]

elif download_all:
    keys = ["bert", "covid_net", "hiexpl_lm", "autoint", "sst", "avazu"]
else:
    raise ValueError


if not os.path.exists(downloads_folder):
    os.makedirs(downloads_folder)

subfolders = next(os.walk(downloads_folder))[1]

for key in keys:
    if key in pretrained_model_ids:
        file_id = pretrained_model_ids[key]
        dest_id = "pretrained_" + key
    elif key in data_ids:
        file_id = data_ids[key]
        dest_id = key + "_data"
    else:
        raise ValueError

    if dest_id in subfolders:
        print(
            "Found "
            + dest_id
            + "/ already in "
            + downloads_folder
            + "/. Skipping download for it."
        )
        continue

    destination = downloads_folder + "/" + dest_id + ".zip"
    download_file_from_google_drive(file_id, destination)

    shutil.unpack_archive(destination, downloads_folder)
    os.remove(destination)
