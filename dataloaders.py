"""Dataset loaders with tiny subsets (40h budget). Captioning: COCO (cached) +
Flickr8k (streamed). Returns dict{train,val,test} of (img224_u8, caption)."""
import os, numpy as np
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from PIL import Image

IMG = 224
N_TRAIN, N_VAL, N_TEST = 200, 50, 100     # tiny per user's 40h constraint


def _prep(pil, img=IMG):
    im = pil.convert("RGB"); w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((img, img), Image.LANCZOS), np.uint8)


COCO_CACHE, COCO_N = "results/coco_cache.npz", 500


def coco_cache():
    """Shared 500-image COCO pool at 224 px (captions: first reference, <=120 chars).
    Streamed once from the COCO test split and cached; every resolution is resized from it."""
    if not os.path.exists(COCO_CACHE):
        from datasets import load_dataset
        ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="test", streaming=True)
        imgs, caps = [], []
        for ex in ds:
            try:
                imgs.append(_prep(ex["jpg"], 224)); caps.append(str(ex["txt"]).strip().split("\n")[0][:120])
            except Exception:
                continue
            if len(imgs) >= COCO_N:
                break
        os.makedirs("results", exist_ok=True)
        np.savez_compressed(COCO_CACHE, imgs=np.stack(imgs), caps=np.array(caps, dtype=object))
    return np.load(COCO_CACHE, allow_pickle=True)


def get_captioning(name, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, img=IMG):
    total = n_train + n_val + n_test
    cache = f"results/cap_{name}_{img}.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True); imgs, caps = list(d["imgs"]), list(d["caps"])
    else:
        imgs, caps = [], []
        if name == "coco":
            d = coco_cache()   # already 224
            imgs, caps = list(d["imgs"])[:total], [str(c) for c in d["caps"]][:total]
        elif name == "flickr8k":
            from datasets import load_dataset
            ds = load_dataset("ariG23498/flickr8k", split="train", streaming=True)
            for ex in ds:
                try:
                    imgs.append(_prep(ex["image"], img)); caps.append(str(ex["caption"]).strip()[:120])
                except Exception:
                    continue
                if len(imgs) >= total:
                    break
        else:
            raise ValueError(name)
        os.makedirs("results", exist_ok=True)
        np.savez_compressed(cache, imgs=np.stack(imgs), caps=np.array(caps, dtype=object))
    pairs = list(zip(imgs, caps))
    return {"train": pairs[:n_train], "val": pairs[n_train:n_train + n_val], "test": pairs[-n_test:]}


if __name__ == "__main__":
    for nm in ["flickr8k", "coco"]:
        d = get_captioning(nm)
        print(nm, "train/val/test:", len(d["train"]), len(d["val"]), len(d["test"]),
              "| sample cap:", d["train"][0][1][:60])
