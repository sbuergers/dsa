from pathlib import Path

from azureml.core import Workspace, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


class AzureCompute:
    """Convenience class for creating or connecting to and managing
    an Azure compute instance and environment.
    
    Parameters
    ----------
    env_yml : str
        Path to .yml file with environment specifications
    compute_name : str
        Name of compute to use (or name for new compute to create)
    vm_size : str (default="STANDARD_DS11_V2")
        Virtual machine size - see
        https://docs.microsoft.com/en-us/azure/virtual-machines/sizes
        https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu
    max_nodes : int (default=2)
        Maximum number of nodes to use
    """
    def __init__(
        self,
        env_yml: str,
        compute_name: str,
        vm_size: str = "STANDARD_DS11_V2",
        max_nodes: int = 2,
    ):
        self.workspace = Workspace.from_config()
        self.environment = self.set_environment(env_yml)
        self.compute = self.get_or_create_compute(compute_name, vm_size, max_nodes)
        
    def get_or_create_compute(
        self,
        compute_name: str,
        vm_size: str = "STANDARD_DS11_V2",
        max_nodes: int = 2,
    ) -> ComputeTarget:
        """Connect to or create new compute instance.

        Parameters
        ----------
        compute_name : str
            Name of compute to use (or name for new compute to create)
        vm_size : str (default="STANDARD_DS11_V2")
            Virtual machine size - see
            https://docs.microsoft.com/en-us/azure/virtual-machines/sizes
            https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu
        max_nodes : int (default=2)
            Maximum number of nodes to use
        
        Returns
        -------
        compute : ComputeTarget
            Azure ML compute instance.
        """
        try:
            compute = ComputeTarget(workspace=self.workspace, name=compute_name)
            print('Found compute "{}", using it.'.format(compute_name))
        except ComputeTargetException:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size, max_nodes=max_nodes
            )
            compute = ComputeTarget.create(self.workspace, compute_name, compute_config)
            compute.wait_for_completion(show_output=True)
            print('Created compute "{}"'.format(compute_name))

        return compute
    
    def set_environment(
        self,
        env_yml: str, 
        env_name: str = None, 
        register_env : bool = True
    ) -> Environment:
        """Set environment based on .yml file.
        
        Parameters
        ----------
        env_yml : str
            Path to .yml file with environment specifications
        env_name : str (default=None)
            When not given the environment will be called as the filename
            (without the extension)
        register_env : bool (default=True)
            Whether to register the environment in the ML workspace
        
        Returns
        -------
        env : Environment
            Environment instance based on .yml file
        """
        if env_name is None:
            env_name = Path(env_yml).stem

        environment = Environment.from_conda_specification(env_name, env_yml)
        
        if register_env:
            environment.register(workspace=self.workspace)

        # Print the environment details
        print(environment.name, 'defined.')
        print(environment.python.conda_dependencies.serialize_to_string())

    def list_environments(self):
        """List all environments registered in the workspace.
        """
        print('Environments registered in workspace {}:\n'.format(self.workspace.name))
        for env in Environment.list(workspace=self.workspace):
            print("Name",env)
