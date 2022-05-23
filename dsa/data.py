from dotenv import load_dotenv
import os
from pprint import pprint

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
        datastore_names: list,
        account_name: str = None,
        container_names: list = None,
        account_key: str = None,
    ):
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
        description: str = "",
        tags : dict = {},
        create_new_version : bool = True,
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
            name=dataset_name,
            description=description,
            tags=tags,
            create_new_version=create_new_version,
        )
