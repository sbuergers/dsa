from typing import Union

from azureml.core import Workspace, Environment, Model, Experiment, Run
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PublishedPipeline
from azureml.pipeline.core.run import PipelineRun

from .compute import AzureCompute
from .data import AzureData


class AzurePipeline(AzureData, AzureCompute):
    """Convenience class for creating azure machine learning pipelines.
    
    Parameters
    ----------
    datastore_names : list
        Name of blob datastores (e.g. {{account_name}}_raw, {{account_name}}_enriched)
    env_yml : str
        Path to .yml file with environment specifications
    account_name : str (default=None)
        Storage account name used for establishing connection when creating new
        datastore (an azure blob store)
    container_names : list (default=None)
        Azure blob container names used to create new datastore connections
        (e.g. raw, enriched)
    account_key : str (default=None)
        Storage account key used to create new datastore connection
    compute_name : str
        Name of compute to use (or name for new compute to create)
    vm_size : str (default="STANDARD_DS11_V2")
        Virtual machine size - see
        https://docs.microsoft.com/en-us/azure/virtual-machines/sizes
        https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu
    max_nodes : int (default=2)
        Maximum number of nodes to use
    
    Examples
    --------
    # Ex. 1
    # This example uses the scripts in dsa/pipeline_steps. For your own pipeline,
    # you should rewrite the define_pipeline_steps method and any associated scripts.
    from dsa.pipeline import ExampleAzurePipeline
    
    datastore_names = ['raw', 'enriched', 'preprocessed']
    env_yml = 'environment.yml'
    account_key = os.environ.get('STORAGE_ACCOUNT_KEY')

    ap = ExampleAzurePipeline(
        datastore_names=datastore_names,
        env_yml=env_yml, 
        container_names=datastore_names,
        account_name='dsablobstore',
        account_key=account_key, 
        compute_name='azdatascientist',
    )
    ap.create_pipeline()
    ap.run_pipeline()
    ap.list_child_run_metrics()
    """
    def __init__(self, **kwargs):
        
        # Could also call super().__init__(self), but it requires absolutely
        # no conflicts between parent classses, so this is safer.
        keys_data = ['datastore_names', 'account_name', 'container_names', 'account_key']
        kwargs_data = {key: kwargs.get(key) for key in keys_data}
        AzureData.__init__(self, **kwargs_data)

        keys_compute = ['env_yml', 'compute_name', 'vm_size', 'max_nodes']
        kwargs_compute = {key: kwargs.get(key) for key in keys_compute}
        AzureCompute.__init__(self, **kwargs_compute)
        
        self.pipeline = None
        self.pipeline_run_config = self.set_pipeline_run_config()
        self.experiment = None
        
    def set_pipeline_run_config(self) -> RunConfiguration:
        """Set pipeline run configuration using current compute and environment.

        Returns
        -------
        RunConfiguration
        """
        pipeline_run_config = RunConfiguration()
        pipeline_run_config.target = self.compute
        pipeline_run_config.environment = self.environment
        print ("Run configuration created.")

        return pipeline_run_config
    
    def define_pipeline_steps(self):
        """This is a placeholder function that needs to filled in for each project.
        This defines all necessary script steps.
        
        See:
        https://github.com/MicrosoftLearning/mslearn-dp100/blob/main/08%20-%20Create%20a%20Pipeline.ipynb
        """
        pass
    
    def create_pipeline(self) -> Pipeline:
        """Combine pipeline steps into a pipeline object.
        
        Returns
        -------
        pipeline: Pipeline
        """
        pipeline_steps = self.define_pipeline_steps()
        if pipeline_steps is None:
            raise ValueError(
                "Please define the method `define_pipeline_steps`"
                ", before running `create_pipeline`"
            )

        pipeline = Pipeline(workspace=self.workspace, steps=pipeline_steps)
        self.pipeline = pipeline

        return pipeline
    
    def get_or_create_experiment(
        self, 
        experiment_name: str = 'default-experiment'
    ) -> Experiment:
        """Get existing azure ml experiment or create new experiment
        
        Parameters
        ----------
        experiment_name : str (default='default-experiment')
        
        Returns
        -------
        experiment : Experiment
            Azure ml experiment instance
        """
        experiment = Experiment(workspace=self.workspace, name=experiment_name)
        self.experiment = experiment

        return experiment

    def run_pipeline(
        self,
        pipeline: Pipeline = None,
        experiment_name: str = None,
        regenerate_outputs: bool = True,
    ) -> PipelineRun:
        """Run the pipeline.
        
        Parameters
        ----------
        pipeline : Pipeline (default=None)
            Azure ML pipeline object to run. When None, run self.pipeline instead.
        regenerate_outputs : bool (default=True)
            Whether to overwrite previously generated outputs or not.
            
        Returns
        -------
        pipeline_run : Run
        """
        if self.experiment is None:
            self.experiment = self.get_or_create_experiment()

        if pipeline is not None:
            pipeline_run = self.experiment.submit(
                pipeline, 
                regenerate_outputs=regenerate_outputs
            )
            pipeline_run.wait_for_completion(show_output=True)
            
        elif self.pipeline is None:
            raise ValueError("No pipeline object has been created yet, nothing to run.")

        else:
            pipeline_run = self.experiment.submit(
                self.pipeline, 
                regenerate_outputs=regenerate_outputs
            )
            pipeline_run.wait_for_completion(show_output=True)

        self.pipeline_run = pipeline_run
        
        return pipeline_run
        
    def publish_pipeline(self, name: str, description: str) -> PublishedPipeline:
        """Publish pipeline. Automatically increases version number.
        
        Parameters
        ----------
        name : str
            Name of to be published pipeline endpoint
        description : str
            Description of to be published pipeline endpoint
            
        Returns
        -------
        published_pipeline : PublishedPipeline
        """
        if self.pipeline is None:
            raise ValueError('No pipeline is defined, cannot publish')

        published_pipelines = PublishedPipeline.list(workspace=self.workspace)
        pipeline_old = None
        version = '1.0'
        for pipe in published_pipelines:
            if name == pipe:
                pipeline_old = pipe
                version = str(np.float(pipeline_old.version) + 1.0)

        published_pipeline = self.pipeline.publish_pipeline(name, description, version)
    
        return published_pipeline
        
    def list_child_run_metrics(self, pipeline_run: PipelineRun = None):
        """List all metrics logged in child runs of a pipeline run.
        
        Parameters
        ----------
        pipeline_run : PipelineRun
            Azure ML PipelineRun instance.
        """
        if pipeline_run is None:
            try:
                pipeline_run = self.pipeline_run
            except:
                print('There is currently no pipeline run defined')
                return

        for run in pipeline_run.get_children():
            print(run.name, ':')
            metrics = run.get_metrics()
            for metric_name in metrics:
                print('\t',metric_name, ":", metrics[metric_name])
    
    def list_registered_models(self):
        """List all models registered in workspace. When a pipeline run has
        successfully completed a model with `Training context` tag should have
        automatically been created.
        """
        for model in Model.list(self.workspace):
            print(model.name, 'version:', model.version)
            for tag_name in model.tags:
                tag = model.tags[tag_name]
                print ('\t',tag_name, ':', tag)
            for prop_name in model.properties:
                prop = model.properties[prop_name]
                print ('\t',prop_name, ':', prop)
            print('\n')


class ExampleAzurePipeline(AzurePipeline):
    
    def define_pipeline_steps(self):

        # Get the training dataset
        diabetes_ds = self.workspace.datasets.get("diabetes-dataset")
        if diabetes_ds is None:
            self.create_and_register_diabetes_dataset()
            diabetes_ds = self.workspace.datasets.get("diabetes-dataset")

        # Create an OutputFileDatasetConfig (temporary Data Reference) for 
        # data passed from step 1 to step 2
        prepped_data = OutputFileDatasetConfig("prepped_data")

        # Step 1, Run the data prep script
        prep_step = PythonScriptStep(
            name = "Prepare Data",
            source_directory = 'dsa/pipeline_steps',
            script_name = "prep_diabetes.py",
            arguments = [
                '--input-data', diabetes_ds.as_named_input('raw_data'),
                '--prepped-data', prepped_data,
            ],
            compute_target = self.compute,
            runconfig = self.pipeline_run_config,
            allow_reuse = True,
        )

        # Step 2, run the training script
        train_step = PythonScriptStep(
            name = "Train and Register Model",
            source_directory = "dsa/pipeline_steps",
            script_name = "train_diabetes.py",
            arguments = ['--training-data', prepped_data.as_input()],
            compute_target = self.compute,
            runconfig = self.pipeline_run_config,
            allow_reuse = True,
        )
        
        return [prep_step, train_step]
