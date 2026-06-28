import os.path
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import kagglehub
import pandas as pd
import torch


from helper.directory_functions import is_dataset_dir_existing, create_dir_name, get_root, \
    search_memotion_dataset_7k_dir, is_memotion_dataset_7k_existing

class ExtractFeaturesKaggle:
    DATASET_DIR = os.path.join(get_root(), "data", "dataset")

    def __init__(self, dataset_name="williamscott701/memotion-dataset-7k"):
        """
        Klasse, die das Laden und im Projektverzeichnis ablegen von
        Datensets aus Keggle kapselt

        :param dataset_name: default="williamscott701/memotion-dataset-7k"
        """
        self.dataset_name = dataset_name
        self.dataset_dir_name = create_dir_name(self.dataset_name)
        self._is_full_dataset_dir_existing = is_dataset_dir_existing(self.dataset_dir_name)
        self._is_memotion_dataset_7k_dir_existing = is_memotion_dataset_7k_existing()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # path = kagglehub.dataset_download("williamscott701/memotion-dataset-7k")
    def load_and_save_dataset(self, dataset_name="williamscott701/memotion-dataset-7k"):
        """
        Läd das Datenset runter, legt ein Überordner an /<Username>_<Datasetname>_

        :param dataset_name: default="williamscott701/memotion-dataset-7k"
        :return: True || False je nachdem ob das Runterladen funktioniert hat
        """
        if not self._is_full_dataset_dir_existing:
            if dataset_name == self.dataset_name:
                dataset_dir = os.path.join(self.DATASET_DIR, self.dataset_dir_name)
                print(f"directory to dataset: {dataset_dir}")
                kagglehub.dataset_download(dataset_name, output_dir=dataset_dir)
                self._is_full_dataset_dir_existing = is_dataset_dir_existing(self.dataset_dir_name)
                return True
            else:
                print(f"{dataset_name} is not this dataset")
                return None
        else:
            print("Dataset is already loaded locally")
            return None

    def load_dataset_from_dir(self, dataset_name="williamscott701/memotion-dataset-7k"):
        """
        Läd den Datensatz aus dem Projektverzeichnis, aktuell sucht es den in load_and_save_dataset() angelegten
        Überordner oder den für das default Datenset übliche Verzeichnis memotion-dataset-7k und läd dort die
        Daten heraus

        :param dataset_name: default="williamscott701/memotion-dataset-7k"
        :return:
        """
        #TODO: der self.get_labels_csv() Aufruf ist irgendwie redundant, da
        # ja der komplette Datenpfad zu memotion_dataset_7k zurückgegeben wird und
        # da ja auch der Überordner drin ist, der in self.load_and_save_dataset
        # generiert wird
        if self._is_memotion_dataset_7k_dir_existing:

            if self._is_full_dataset_dir_existing:
                if dataset_name == self.dataset_name:
                    csv_data = pd.read_csv(self.get_labels_csv_path())
                    return csv_data
                else:
                    print(f"{dataset_name} is not this dataset")
                    return None
            else:
                csv_data = pd.read_csv(self.get_labels_csv_path())
                return csv_data
        else:
            print("Dataset is not loaded locally")
            return None

    def is_dataset_loaded_locally(self) -> bool:
        """
        Getter für die Zustandsvariable die nachprüft, ob das in load_and_save_dataset()
        angelegte Überverzeichnis existiert

        :return:
        """
        return self._is_full_dataset_dir_existing

## ----- GETTER ------------------------------------------------------

    def get_images_path(self) -> str:
        """
        gibt den kompletten Datenpfad zu /images zurück, beginnend bei Laufwerk C
        :return:
        """
        memotion_dataset_7k_dir = search_memotion_dataset_7k_dir()
        if memotion_dataset_7k_dir is not None:
            return os.path.join(memotion_dataset_7k_dir, "images")
        else:
            self.load_and_save_dataset()
            return self.get_images_path()

    def get_labels_csv_path(self) -> str:
        """
        gibt den kompletten Datenpfad zu labels.csv zurück, beginnend bei Laufwerk C
        :return:
        """
        memotion_dataset_7k_dir = search_memotion_dataset_7k_dir()
        if memotion_dataset_7k_dir is not None:
            return os.path.join(memotion_dataset_7k_dir, "labels.csv")
        else:
            self.load_and_save_dataset()
            return self.get_images_path()