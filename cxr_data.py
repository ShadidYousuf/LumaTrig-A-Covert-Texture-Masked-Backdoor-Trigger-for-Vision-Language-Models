"""Cache a balanced chest X-ray set (normal vs pneumonia) at 512px for the medical
backdoor demo. No token needed. -> results/cxr_512.npz {imgs, labels}."""
import os, numpy as np
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from PIL import Image

SIZE, PER_CLASS = 512, 200
CACHE = "results/cxr_512.npz"


def prep(pil):
    im = pil.convert("RGB"); w, h = im.size; s = min(w, h)
    im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(im, np.uint8)


def main():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print("cached:", len(d["imgs"]), "labels", dict(zip(*np.unique(d["labels"], return_counts=True))))
        return
    from datasets import load_dataset
    ds = load_dataset("hf-vision/chest-xray-pneumonia", split="test")   # 624 imgs, both classes
    imgs, labs, cnt = [], [], {0: 0, 1: 0}
    for ex in ds:
        y = int(ex["label"])
        if cnt.get(y, 0) >= PER_CLASS:
            continue
        try:
            imgs.append(prep(ex["image"])); labs.append(y); cnt[y] = cnt.get(y, 0) + 1
        except Exception:
            continue
        if all(cnt.get(c, 0) >= PER_CLASS for c in (0, 1)):
            break
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(CACHE, imgs=np.stack(imgs), labels=np.array(labs))
    print(f"cached {len(imgs)} CXR @ {SIZE}px; label counts {cnt}  (0=NORMAL,1=PNEUMONIA)")


if __name__ == "__main__":
    main()
