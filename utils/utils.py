import os
from pathlib import Path
import shutil



def create_tmp_dir(path):
    """
    Create the output directory if NOT exist,
    or Remove the previous result and recreate the one.
    
    Args:
        path: str
            directory PATH
    example
    -------
    path = './work' --> './work/output/'
    """
    # Result output folder: create if NOT exist
    OUTPUT_PATH = Path(path) / 'output'
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    else:
        # remove the previous results
        shutil.rmtree(OUTPUT_PATH)
        os.makedirs(OUTPUT_PATH)



if __name__ == '__main__':
    cwd_path = os.path.dirname(__file__)
    create_tmp_dir(cwd_path)