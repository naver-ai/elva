# Elva
# Copyright (c) 2024-present NAVER Cloud Corp.
# MIT license
import json
from PIL import Image
from pathlib import Path
import lmdb
import fire
from io import BytesIO
from tqdm import trange

LMDB_MAP_SIZE = 1024**4  # 1 TiB

def main(
    metadata_jsonl = "path/to/metadata.json",
    image_folder = "path/to/images",
    lmdb_path = "path/to/lmdb",
    pid = 0,
    pnum = 1,
):
    dataset = json.load(open(metadata_jsonl))
    print(len(dataset))
    print(pid)
    print(pnum)
    chunk_idxes = list(range(0, len(dataset), int(len(dataset) / pnum) + 1)) + [len(dataset)]
    s, e = list(zip(chunk_idxes[:-1], chunk_idxes[1:]))[pid]
    print(s, e)

    image_lmdb = lmdb.open(
        lmdb_path,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_size=LMDB_MAP_SIZE,
        max_readers=256,
    )

    for sample_idx in trange(s, e):
        if 'image' in dataset[sample_idx]:
            image_file = BytesIO()
            Image.open(Path(image_folder) / dataset[sample_idx]['image']).save(image_file, format="JPEG")
            image_bytes = image_file.getvalue()
            image_lmdb_txn = image_lmdb.begin(write=True)
            image_lmdb_txn.put(dataset[sample_idx]['image'].encode("utf-8"), image_bytes)
            image_lmdb_txn.commit()

    image_lmdb.close()

if __name__ == "__main__":
    fire.Fire(main)