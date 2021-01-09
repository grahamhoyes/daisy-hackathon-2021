#!/usr/bin/env python3

import requests
import json
import os
from zipfile import ZipFile
import argparse

DEFAULT_SERVER = "34.95.62.180"


def submit(token, import_string, server=DEFAULT_SERVER, filenames=[]):
    params = {
        "token": token,
        "import_string": import_string,
    }

    zipname = "./submission.zip"
    with ZipFile(zipname, "w") as z:
        for filename in filenames:
            z.write(filename)

    files = {
        "params": (None, json.dumps(params), "application/json"),
        "submission": (
            os.path.basename(zipname),
            open(zipname, "rb"),
            "application/octet-stream",
        ),
    }

    url = f"http://{server}/submit"
    print("Submitting to server...")
    r = requests.post(url, files=files)
    print(f"{r.status_code} - {r.text}")


def main():
    parser = argparse.ArgumentParser(
        description="Daisy Hackathon 2021 submission script"
    )

    parser.add_argument(
        "--server",
        type=str,
        required=False,
        default=DEFAULT_SERVER,
        help="List of files needed to run your player class",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="team token provided by Daisy, used to identify your team",
    )
    parser.add_argument(
        "--player-class",
        type=str,
        required=True,
        help="<module>:<classname> string of your SiteLocationPlayer class",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="List of files needed to run your player class",
    )

    args = parser.parse_args()

    submit(args.token, args.player_class, filenames=args.files, server=args.server)


if __name__ == "__main__":
    main()
