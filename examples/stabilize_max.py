from pathlib import Path

from selfie_stabilizer import stabilize_selfies

IMAGE_DIR = Path("data")
FN_REF = "data/Max-186.jpg"
OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FILE_PATTERN = "Max-*.jpg"
MOVIE_FN = "MaxAligned.mp4"


if __name__ == "__main__":
    stabilize_selfies(IMAGE_DIR, FN_REF, vid_fn=MOVIE_FN, img_out_dir=OUT_DIR)
