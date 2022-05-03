"""
https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
"""
from pathlib import Path
import numpy as np
import dlib
from tqdm import tqdm
import imutils
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from icecream import ic
import itertools
from multiprocessing import Pool

WIDTH = 1000

IMAGE_DIR = Path("data")
FN_REF = "data/Max-186.jpg"
OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FILE_PATTERN = "Max-*.jpg"

fn_shape_predictor = Path("shape_predictor_68_face_landmarks.dat")
assert fn_shape_predictor.exists()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(fn_shape_predictor.as_posix())


def process_dlib_faces(rects, center_x, center_y):
    """
    finds the main face. returns main face and all the other faces as dlib.Rectangles
    """
    if len(rects) == 0:
        return None, None
    x = np.array([rect.center().x for rect in rects])
    y = np.array([rect.center().y for rect in rects])

    i = np.argmin((center_x - x) ** 2 + (center_y - y) ** 2)

    main = rects.pop(i)

    return main, rects


def plot(image, main_rect=None, other_rects=None, landmarks=None, save=None):
    fig = plt.figure()
    ax = plt.gca()
    plt.imshow(image)

    if main_rect:
        ax.add_patch(
            Rectangle(
                (main_rect.left(), main_rect.top()),
                main_rect.width(),
                main_rect.height(),
                edgecolor="red",
                facecolor="none",
                lw=4,
            )
        )
    if other_rects:
        for rect in other_rects:
            ax.add_patch(
                Rectangle(
                    (rect.left(), rect.top()),
                    rect.width(),
                    rect.height(),
                    edgecolor="green",
                    facecolor="none",
                    lw=1,
                )
            )

    if landmarks is not None:
        x = landmarks[:, 0]
        y = landmarks[:, 1]
        plt.plot(x, y, "w.", ms=2)

    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()


def get_landmarks(image):
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    rects = detector(image)
    main_rect, other_rects = process_dlib_faces(rects, center_x, center_y)

    try:
        landmarks = np.array([(p.x, p.y) for p in predictor(image, main_rect).parts()])
    except:
        landmarks = None

    return landmarks, main_rect, other_rects


def align_landmarks(landmarks, ref):
    ref = np.array(ref, dtype=float)
    cx, cy = landmarks.mean(axis=0)
    cx_ref, cy_ref = ref.mean(axis=0)

    # sx, sy = landmarks.std(axis=0)
    # sx_ref, sy_ref = ref.std(axis=0)

    out = np.array(landmarks, dtype=float)

    out[:, 0] -= cx
    out[:, 1] -= cy

    s = out.std()
    out /= s

    ref[:, 0] -= cx_ref
    ref[:, 1] -= cy_ref

    s_ref = ref.std()
    ref /= s_ref

    H = ref.T @ out

    S, _, Vt = np.linalg.svd(H)

    R_ = S @ Vt

    out = (R_ @ out.T).T
    out *= s_ref
    out[:, 0] += cx_ref
    out[:, 1] += cy_ref

    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    S1 = np.array([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1]])
    R = np.eye(3)
    R[:2, :2] = R_
    S2 = np.array([[s_ref, 0, 0], [0, s_ref, 0], [0, 0, 1]])
    T2 = np.array([[1, 0, cx_ref], [0, 1, cy_ref], [0, 0, 1]])

    A = T2 @ S2 @ R @ S1 @ T1

    return A, out


def run_single(fn, landmarks_ref):
    fn_out = OUT_DIR / (Path(fn).stem + ".png")
    image = cv2.imread(fn)
    image = imutils.resize(image, width=1000)
    rows, cols = image.shape[:2]
    # print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks, main_rect, other_rects = get_landmarks(gray)
    if landmarks is not None:
        A, landmarks_aligned = align_landmarks(landmarks, landmarks_ref)
        aligned = cv2.warpAffine(image, A[:2, :], (cols, rows))
    else:
        return

    aligned_numpy = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    plot(aligned_numpy, save=fn_out)


def run_single_multi(x):
    return run_single(x[0], x[1])


def run_serial(file_list, landmarks_ref):
    for fn in tqdm(file_list):
        run_single(fn, landmarks_ref)


def run_par(file_list, landmarks_ref):
    params = zip(file_list, itertools.repeat(landmarks_ref))
    with Pool() as pool:
        for _ in tqdm(pool.imap(run_single_multi, params), total=len(file_list)):
            pass


def rename_image_files(directory):
    file_list = list(sorted(Path(directory).iterdir()))

    for i, file in enumerate(file_list):
        file.rename(Path(directory) / f"{i:03d}.png")


if __name__ == "__main__":
    file_list = [x.as_posix() for x in IMAGE_DIR.glob(FILE_PATTERN)]
    file_list.sort()

    # reference landmarks
    image = cv2.imread(FN_REF)
    image = imutils.resize(image, width=1000)
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks_ref, _, _ = get_landmarks(gray)

    # run_serial(file_list, landmarks_ref)
    run_par(file_list, landmarks_ref)
    rename_image_files("out/")
