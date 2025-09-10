import os
import slideio
from multiprocessing import Pool
from tqdm import tqdm

params = slideio.SVSJpegParameters()
params.quality = 30


def convert_czi_to_svs(args):
    czi_path, output_dir = args
    try:
        slide_czi = slideio.open_slide(czi_path)
        scene = slide_czi.get_scene(0)
        output_filename = os.path.join(output_dir, f"{os.path.basename(czi_path).replace('.czi', '')}.svs")
        slideio.convert_scene(scene, params, output_filename)
        del slide_czi
        return True
    except Exception as e:
        print(f"Failed: {czi_path}, Error: {str(e)}")
        return False


def batch_convert(czi_dir, svs_dir):
    os.makedirs(svs_dir, exist_ok=True)
    czi_files = [f for f in os.listdir(czi_dir) if f.lower().endswith(".czi")]
    tasks = [(os.path.join(czi_dir, f), svs_dir) for f in czi_files]

    with Pool(processes=6) as pool:  # 使用4个进程
        results = list(tqdm(pool.imap(convert_czi_to_svs, tasks), total=len(tasks), desc="Converting"))

    success_count = sum(results)
    print(f"Completed! Success: {success_count}/{len(czi_files)}")


if __name__ == "__main__":
    czi_dir = r"/media/public506/sdb/SY/sr386/SR86"
    svs_dir = r"/media/public506/sdb/SY/sr386/SVS"
    batch_convert(czi_dir, svs_dir)