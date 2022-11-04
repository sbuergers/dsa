from dotenv import load_dotenv
import os
from pathlib import Path, PosixPath
from pprint import pprint
import pandas as pd

from azureml.core import Datastore, Workspace, Dataset
from azureml.exceptions import UserErrorException

load_dotenv()


class AzureData:
    """Base class for connecting to azure blob storage container. This class is
    wrapping several common methods from preparing datasets for azure machine learning.
    See:
    https://github.com/MicrosoftLearning/mslearn-dp100/blob/main/06%20-%20Work%20with%20Data.ipynb

    Parameters
    ----------
    datastore_names : list
        Name of blob datastores (e.g. {{account_name}}_raw, {{account_name}}_enriched)
    account_name : str (default=None)
        Storage account name used for establishing connection when creating new
        datastore (an azure blob store)
    container_names : list (default=None)
        Azure blob container names used to create new datastore connections
        (e.g. raw, enriched)
    account_key : str (default=None)
        Storage account key used to create new datastore connection
    """
    def __init__(
        self,
        datastore_names: list = ['raw', 'preprocessed', 'enriched'],
        account_name: str = None,
        container_names: list = ['raw', 'preprocessed', 'enriched'],
        account_key: str = None,
    ):
        self.root_path = Path(__file__).parent.parent
        self.workspace = Workspace.from_config()
        if account_key is None:
            account_key = os.environ.get('STORAGE_ACCOUNT_KEY')

        self.datastores = {}
        self.blob_services = {}
        for datastore_name, container_name in zip(datastore_names, container_names):
            self.datastores[datastore_name] = self.get_or_create_datastore(
                datastore_name,
                account_name,
                container_name,
                account_key,
            )
            self.blob_services[datastore_name] = self.datastores[datastore_name].blob_service

            existing_containers = [
                c.name for c in self.blob_services[datastore_name].list_containers()
            ]
            if container_name not in existing_containers:
                self.blob_services[datastore_name].create_container(container_name)

    def get_or_create_datastore(
        self,
        datastore_name: str,
        account_name: str = None,
        container_name: str = None,
        account_key: str = None,
    ) -> Datastore:
        """Establish datastore connection or register new datastore if it does not
        yet exist.

        Parameters
        ----------
        datastore_name : str
            Name of datastore in azure ML
        account_name : str (default=None)
            Storage account name used for establishing connection when creating new
            datastore
        container_name : str (default=None)
            Azure blob container name used to create new datastore connection
        account_key : str (default=None)
            Storage account key used to create new datastore connection

        Returns
        -------
        datastore : azureml.core.datastore.Datastore
        """
        try:
            datastore = Datastore.get(self.workspace, datastore_name)
            print("Found Blob Datastore with name: {}".format(datastore_name))

        except UserErrorException:
            datastore = Datastore.register_azure_blob_container(
                workspace=self.workspace,
                datastore_name=datastore_name,
                account_name=account_name,  # Storage account name
                container_name=container_name,  # Name of Azure blob container
                account_key=account_key,  # Storage account key
            )

        return datastore

    def list_datastores(self):
        """Convenience method for listing all datastores in the class and workspace.
        """
        print('Datastores in AzureData class')
        print('-----------------------------')
        pprint(self.datastores)
        print('\nDatastores in workspace')
        print('-----------------------')
        default_ds = self.workspace.get_default_datastore()
        for ds_name in self.workspace.datastores:
            print(ds_name, "- Default =", ds_name == default_ds.name)

    def get_tabular_dataset(self, datastore_name: str, data_path: str) -> Dataset:
        """Get tabular dataset from data_path from Datastore from storage account.

        Parameters
        ----------
        datastore_name : str
            Name of azure ML datastore
        data_path : str
            Path of blobs we are interested in. Can be multiple .csv files in one
            folder, e.g. 'some/directory/*.csv'

        Returns
        -------
        Dataset
            Dataset instance with connection to data blob.
        """
        return Dataset.Tabular.from_delimited_files(
            path=(self.datastores.get(datastore_name), data_path)
        )

    def get_file_dataset(self, datastore_name: str, data_path: str) -> Dataset:
        """Get file dataset from data_path from Datastore from storage account.

        Parameters
        ----------
        datastore_name : str
            Name of azure ML datastore
        data_path : str
            Path of blobs we are interested in. Can be multiple .csv files in one
            folder, e.g. 'some/directory/*.csv'

        Returns
        -------
        Dataset
            Dataset instance with connection to data blob.
        """
        return Dataset.Files.from_files(
            path=(self.datastores.get(datastore_name), data_path)
        )

    def upload_files(
        self,
        datastore_name: str,
        files: list,
        target_path: str,
        overwrite: bool = True,
        show_progress: bool = False,
    ):
        """Upload `files` to `data_path` to blob store container connected to
        `datastore_name` (wrapper for {{datastore}}.upload_files())

        Parameters
        ----------
        datastore_name : str
            Name of datastore in azure ML
        files : list
            List of local paths of files (e.g. ['./myfile1.csv', './myfile2.csv'])
        target_path : str
            Directory on blob store to save to (e.g. 'mydata/')
        overwrite : bool (default=True)
            Wether to overwrite files if they already exist
        show_progress : bool (default=False)
            Wether to print progress
        """
        self.datastores.get(datastore_name).upload_files(
            files=files,
            target_path=target_path,
            overwrite=overwrite,
            show_progress=show_progress,
        )

    def register_dataset(
        self,
        dataset: Dataset,
        name: str,
        description: str = "",
        tags: dict = {},
        create_new_version: bool = True,
    ):
        """Register specific dataset with machine learning workspace for easy
        use later on. Wrapper for Dataset.register using the current workspace.

        NOTE: This is not strictly necessary, but can be convient when multiple
        scripts use the same dataset. The versioning can also be useful to track
        which version of a dataset was used for obtaining specific results.

        Parameters
        ----------
        dataset : Dataset
            Dataset to register with machine learning workspace (either tabular or
            file based).
        name : str
            Name of the dataset
        description : str (default="")
            Short description of dataset
        tags : dict (default={})
            Tags can be changed, descriptions are fixed
        create_new_version : bool (default=True)
            Whether to create a new version of this dataset if a previous version
            already is registered in the workspace
        """
        dataset.register(
            workspace=self.workspace,
            name=name,
            description=description,
            tags=tags,
            create_new_version=create_new_version,
        )

    def create_and_register_diabetes_dataset(self, root_folder: PosixPath = Path('')):
        """This method retrieves the diabetes dataset from the microsoft azure ml
        tutorials and registers it in the current workspace. This can be useful for
        running tests or trying out the base classes of this repository. The dataset
        will be called 'diabetes-dataset' and registered as such.

        Parameters
        ----------
        root_folder : PosixPath (default=Path(''))
            Folder that contains the `data` folder
        """
        if 'raw' not in self.datastores.keys():
            raise ValueError("This method only works when a 'raw' datastore is present.")

        os.makedirs(root_folder / 'data', exist_ok=True)
        diabetes = pd.read_csv(
            'https://github.com/MicrosoftLearning/mslearn-dp100/blob/main/data/diabetes.csv?raw=true'  # noqa: E501
        )
        diabetes.to_csv(root_folder / Path('data/diabetes.csv'))
        diabetes2 = pd.read_csv(
            'https://github.com/MicrosoftLearning/mslearn-dp100/blob/main/data/diabetes2.csv?raw=true'  # noqa: E501
        )
        diabetes2.to_csv(root_folder / Path('data/diabetes2.csv'))

        self.upload_files(
            datastore_name='raw',
            files=[
                str(root_folder / Path('data/diabetes.csv')),
                str(root_folder / Path('data/diabetes2.csv'))
            ],
            target_path='diabetes_data/',
        )

        diabetes_dataset = self.get_tabular_dataset(
            datastore_name='raw',
            data_path='diabetes_data/*.csv',
        )
        self.register_dataset(diabetes_dataset, name='diabetes-dataset')
