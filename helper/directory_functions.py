import os


def is_dataset_dir_existing(dataset_dir_name: str) -> bool:
    """
    checks if dataset directory (dataset_dir_name) exists in "./data/dataset"
    """
    root = get_root()

    # get complete path to dataset
    dataset_path = os.path.join("data", "dataset")
    searched_dir = os.path.join(root, dataset_path)
    if os.listdir(searched_dir).count(dataset_dir_name) == 1:
        return True
    else:
        return False


def get_root() -> str:
    """get project root"""
    project_name = "EMI_Project"
    cwd = os.getcwd()
    cwd_list = cwd.split(project_name)
    root = os.path.join(cwd_list[0], project_name)

    return root


def create_dir_name(dataset_name):
    """
    Takes name of dataset, turns it into snake_case directory name

    e.g.:
    "Leonardo6/memotion" -> Leonardo6_memotion_
    """
    name_list = dataset_name.split("/")
    dir_name = ""
    for items in name_list:
        dir_name += (items + "_")

    return dir_name


def search_memotion_dataset_7k_dir():
    target = "memotion_dataset_7k"
    ret_val = search_dir(target)
    if ret_val is not None:
        return ret_val
    return None


def search_dir(searched_dir):
    root = get_root()

    for current_dir, dirs, files in os.walk(root):
        if searched_dir in dirs:
            path = os.path.join(current_dir, searched_dir)
            return path

    return None


def is_memotion_dataset_7k_existing() -> bool:
    if search_memotion_dataset_7k_dir() is not None:
        return True
    else:
        return False
