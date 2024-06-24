import os, shutil, pathlib

screenshot_bp = './screenshots'
anno_bp = './annotations'

def reindex_screenshot():
    parent_dir = pathlib.Path(screenshot_bp)
    for subdir in parent_dir.iterdir():
        if subdir.is_dir():
            for file_path in subdir.iterdir():
                if file_path.is_file():
                    shutil.move(str(file_path), str(parent_dir / file_path.name))
            print(f'{str(subdir)} ok.')
            subdir.rmdir()
        



if __name__ == '__main__':
    reindex_screenshot()