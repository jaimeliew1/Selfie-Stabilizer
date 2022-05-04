import shutil
import tempfile
from pathlib import Path

import click

from . import lib


def stabilize_selfies(
    in_dir, ref_image, vid_fn=None, img_out_dir=None, par=True, show_landmarks=False
):
    file_list = [x.as_posix() for x in Path(in_dir).glob("*")]
    file_list.sort()
    print(f"{len(file_list)} files found in {in_dir}")

    with tempfile.TemporaryDirectory() as temp_dir:

        landmarks_ref = lib.get_reference_landmarks(ref_image)

        print(f"Stabilizing selfies...")
        if par:
            lib.run_par(file_list, landmarks_ref, temp_dir, landmarks=show_landmarks)
        else:
            lib.run(file_list, landmarks_ref, temp_dir, landmarks=show_landmarks)

        lib.rename_image_files(temp_dir)

        if vid_fn:
            print(f"Saving video to {vid_fn}...")
            lib.write_video(temp_dir + "/*.png", vid_fn)
        if img_out_dir:
            print(f"dumping images to {img_out_dir}...")
            shutil.copytree(temp_dir, img_out_dir, dirs_exist_ok=True)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("ref_image", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default="out.mp4", help="Filename of output video.")
@click.option("--dump_images", is_flag=False, flag_value="dump", default=None, help="Save stabilized images to `dump` folder.")
@click.option("-p", "--parallel", is_flag=True, default=True, help="Use parallel processing.")
@click.option("--show_landmarks", default=False, help="Draw facial landmarks.")
def cli(input_dir, ref_image, output, dump_images, parallel, show_landmarks):
    """
    \b
     ____       _  __ _        ____  _        _     _ _ _
    / ___|  ___| |/ _(_) ___  / ___|| |_ __ _| |__ (_) (_)_______ _ __
    \___ \ / _ \ | |_| |/ _ \ \___ \| __/ _` | '_ \| | | |_  / _ \ '__|
     ___) |  __/ |  _| |  __/  ___) | || (_| | |_) | | | |/ /  __/ |
    |____/ \___|_|_| |_|\___| |____/ \__\__,_|_.__/|_|_|_/___\___|_|

    
    Converts a folder of selfie images into a stabilized video.
    """
    stabilize_selfies(
        input_dir, ref_image, output, dump_images, parallel, show_landmarks
    )
