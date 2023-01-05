import os
import shutil



def create_tmp_dir(path):
    """
    Create the directory if NOT exist,
    or Remove the previous result and recreate the one.
    
    Args:
        path: pathlib.Path
            directory PATH to be created
    """
    # Result the folder: create if NOT exist
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # remove the previous results
        shutil.rmtree(path)
        os.makedirs(path)



if __name__ == '__main__':
    cwd_path = os.path.dirname(__file__)
    create_tmp_dir(cwd_path)