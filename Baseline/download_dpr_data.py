# Credits: The design of the source code follows from the DevSinghSachans download_data.py script in his unsupervised-passage-reranking repo 
# Which itself is based on the original DPR repo

"""
 Command line tool to download various preprocessed data sources
"""
import argparse
import tarfile
import os
import pathlib
from subprocess import Popen, PIPE


RESOURCES_MAP = {
    # Wikipedia
    "data.wikipedia-split.psgs_w100": {
        "dropbox_url": "https://www.dropbox.com/s/bezryc9win2bha1/psgs_w100.tar.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },

 

    # DPR data
    "data.retriever-outputs.dpr.entity-questions": {
        "dropbox_url": "https://www.dropbox.com/s/2ngexghb2zzjdie/entity-questions.tar.gz",
        "original_ext": "",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Entity Questions test set.",
    },
    "data.retriever-outputs.dpr.nq-train": {
        "dropbox_url": "https://www.dropbox.com/s/6g4erof4ifg8xea/nq-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Natural Questions Open train set.",
    },
    "data.retriever-outputs.dpr.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/257quanu64w9sh0/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.dpr.reranked.nq-dev": {
        "dropbox_url": "https://www.dropbox.com/s/osolohjruv3dw2y/nq-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the Natural Questions Open development set.",
    },
    "data.retriever-outputs.dpr.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/c7ooi5fgy658cri/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.dpr.reranked.nq-test": {
        "dropbox_url": "https://www.dropbox.com/s/x1nxpf0uz5lapz6/nq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the Natural Questions Open test set.",
    },
    "data.retriever-outputs.dpr.trivia-train": {
        "dropbox_url": "https://www.dropbox.com/s/3onjkogwkc2gk4u/trivia-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the TriviaQA train set.",
    },
    "data.retriever-outputs.dpr.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/50wx42yquqvbbgx/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the TriviaQA test set.",
    },
    "data.retriever-outputs.dpr.reranked.trivia-test": {
        "dropbox_url": "https://www.dropbox.com/s/s7g76bkftwinozw/trivia-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the TriviaQA test set.",
    },
    "data.retriever-outputs.dpr.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/7t7czgqmxyz1ddt/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the TriviaQA dev set.",
    },
    "data.retriever-outputs.dpr.reranked.trivia-dev": {
        "dropbox_url": "https://www.dropbox.com/s/zz3btm8bhaw1c7c/trivia-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the TriviaQA dev set.",
    },
    "data.retriever-outputs.dpr.webq-test": {
        "dropbox_url": "https://www.dropbox.com/s/n8v2o00231e9lkl/webq-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the WebQuestions test set.",
    },
    "data.retriever-outputs.dpr.squad1-train": {
        "dropbox_url": "https://www.dropbox.com/s/i4loxz4k1squ3az/squad1-train.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Squad1-Open train set.",
    },
    "data.retriever-outputs.dpr.squad1-dev": {
        "dropbox_url": "https://www.dropbox.com/s/0r8k4cqtt61ep3e/squad1-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Squad1-Open dev set.",
    },
    "data.retriever-outputs.dpr.reranked.squad1-dev": {
        "dropbox_url": "https://www.dropbox.com/s/tbbm9s1jksw31fk/squad1-dev.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the Squad1-Open dev set.",
    },
    "data.retriever-outputs.dpr.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/91vf2nqmzfvyyx7/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR for the Squad1-Open test set.",
    },
    "data.retriever-outputs.dpr.reranked.squad1-test": {
        "dropbox_url": "https://www.dropbox.com/s/taitdxquvhqc0da/squad1-test.tar.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Top-1000 passages from DPR + UPR (T0-3B) for the Squad1-Open test set.",
    },

   
   
}


def unpack(tar_file: str, out_path: str):
    print("Uncompressing %s", tar_file)
    input = tarfile.open(tar_file, "r:gz")
    input.extractall(out_path)
    input.close()
    print(" Saved to %s", out_path)


def download_resource(
    dropbox_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str
) -> None:
    print("Requested resource from %s", dropbox_url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    print("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, "downloads", *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    print("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        print("File already exist %s", local_file_uncompressed)
        return

    local_file = os.path.abspath(os.path.join(save_root, path_names[-1] + (".tar.gz" if compressed else original_ext)))

    process = Popen(['wget', dropbox_url, '-O', local_file], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))
    # print(stderr.decode("utf-8"))
    print("Downloaded to %s", local_file)

    if compressed:
        # uncompressed_path = os.path.join(save_root, path_names[-1])
        unpack(local_file, save_root)
        os.remove(local_file)
    return



def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        print("matched by prefix resources: %s", resources)
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            print("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    dropbox_url = download_info["dropbox_url"]

    if isinstance(dropbox_url, list):
        for i, url in enumerate(dropbox_url):
            download_resource(
                url,
                download_info["original_ext"],
                download_info["compressed"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
    else:
        download_resource(
            dropbox_url,
            download_info["original_ext"],
            download_info["compressed"],
            resource_key,
            out_dir,
        )
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print("Resource key=%s  :  %s", k, v["desc"])


if __name__ == "__main__":
    main()
