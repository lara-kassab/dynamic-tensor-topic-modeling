"""Allow for reproducibility of results."""
import datetime  # datetime.datetime.now
import os  # os.path
import shutil

from config import results_dir


def log_output(output_name, results_dir=results_dir):
    """Create output folder in the results folder containing output and code.

    Copies the current state of the code in the src and scripts folders to the
    results folder indicated in config.py.

    Args:
        output_name (str): Descriptive name to add to output folder.
        results_dir (str): Alternative results directory to use.

    Returns:
        (str): Path to output folder in which to save experiment output.
    """
    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Get paths to src and scripts folders. Paths are relative to THIS file.
    cur_filepath = os.path.abspath(__file__)
    path_to_covid19 = os.path.abspath(os.path.join(cur_filepath, os.pardir, os.pardir))
    path_to_src = os.path.join(path_to_covid19, "src")
    path_to_scripts = os.path.join(path_to_covid19, "scripts")
    folders_to_copy = [path_to_src, path_to_scripts]

    # Make folder to store results and code snapshot.
    output_folder = "_".join([date_time, output_name])
    dirpath = os.path.join(results_dir, output_folder)
    os.makedirs(dirpath)

    # Ignore hidden files.
    hidden_files = shutil.ignore_patterns(".*", "_*")

    # Save a copy of the current src and scripts folders.
    for folder_path, folder_name in zip(folders_to_copy, ["src", "scripts"]):
        shutil.copytree(
            folder_path, os.path.join(dirpath, "code", folder_name), ignore=hidden_files
        )

    # Make results subfolder.
    dirpath = os.path.join(dirpath, "output")
    os.makedirs(dirpath)

    return dirpath
