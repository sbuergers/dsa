from dotenv import load_dotenv
import os

from azureml.core import Datastore, Workspace
from azureml.exceptions import UserErrorException

load_dotenv()


class AzureDataContainer:
    """Base class for connecting to azure blob storage container.
    
    Parameters
    ----------
    blob_datastore_name : str
        Name of blob datastore
    account_name : str (default=None)
        Storage account name used for establishing connection when creating new
        datastore
    container_name : str (default=None)
        Azure blob container name used to create new datastore connection
    account_key : str (default=None)
        Storage account key used to create new datastore connection
    auth : AzureCliAuthentication
        Authentication object using command line authentication
    """
    def __init__(
        self,
        blob_datastore_name: str,
        account_name: str = None,
        container_name: str = None,
        account_key: str = None,
        auth: AzureCliAuthentication = None,
    ):
        if auth is not None:
            self.workspace = Workspace.from_config(auth=auth)
        else:
            self.workspace = Workspace.from_config()

        self.storage_account = blob_datastore_name
        if account_key is None:
            account_key = os.environ.get('STORAGE_ACCOUNT_KEY')
        if account_name is None:
            datastore_name = '{}_{}'.format(blob_datastore_name, container_name)

        self.datastore = self.get_datastore(
            blob_datastore_name, account_name, container_name, account_key
        )
        self.blob_service = self.datastore.blob_service

    def get_datastore(
        self,
        blob_datastore_name: str,
        account_name: str = None,
        container_name: str = None,
        account_key: str = None,
    ) -> Datastore:
        """Establish datastore connection or register new datastore if it does not
        yet exist.
        
        Parameters
        ----------
        blob_datastore_name : str
            Name of blob datastore
        account_name : str (default=None)
            Storage account name used for establishing connection when creating new
            datastore
        container_name : str (default=None)
            Azure blob container name used to create new datastore connection
        account_key : str (default=None)
            Storage account key used to create new datastore connection
        
        Returns
        -------
        blob_datastore : azureml.core.datastore.Datastore
        """
        try:
            datastore = Datastore.get(self.workspace, blob_datastore_name)
            print("Found Blob Datastore with name: {}".format(blob_datastore_name))

        except UserErrorException:
            datastore = Datastore.register_azure_blob_container(
                workspace=self.workspace,
                datastore_name=blob_datastore_name,
                account_name=account_name,  # Storage account name
                container_name=container_name,  # Name of Azure blob container
                account_key=account_key,  # Storage account key
            )

        return datastore

    def get_dataset(self, data_path: str) -> Dataset:
        """Get tabular dataset from data_path from Datastore from storage account.

        Parameters
        ----------
        data_path : str
            Path of container/blob we are interested in. This is a .csv file.
            
        Returns
        -------
        Dataset
            azureml.core.dataset.Dataset instance with connection to data blob.
        """
        return Dataset.Tabular.from_delimited_files(
            path=(self.datastore, data_path)
        )    
