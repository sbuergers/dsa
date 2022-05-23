from typing import Union

from azureml.core import Workspace, Environment, Model, Run
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PublishedPipeline
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
    """
    def __init__(self, **kwargs):
        
        # Could also call super().__init__(self), but it requires absolutely
        # no conflicts between parent classses, so this is safer.
        AzureData.__init__(self, **kwargs)
        AzureCompute.__init__(self, **kwargs)
        
        self.pipeline = None
        self.pipeline_run_config = set_pipeline_run_config()
        
        
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
            raise(
                "Please define the method `define_pipeline_steps`"
                ", before running `create_pipeline`"
            )

        pipeline = Pipeline(workspace=self.workspace, steps=pipeline_steps)
        self.pipeline = pipeline

        return pipeline
    
    def run_pipeline(
        self,
        pipeline: Pipeline = None,
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
        if pipeline is not None:
            pipeline_run = self.experiment.submit(
                pipeline, 
                regenerate_outputs=regenerate_outputs
            )
        elif self.pipeline is None:
            raise("No pipeline object has been created yet, nothing to run.")
        else:
            pipeline_run = self.experiment.submit(
                self.pipeline, 
                regenerate_outputs=regenerate_outputs
            )

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
        published_pipelines = PublishedPipeline.list(workspace=self.workspace)
        pipeline_old = None
        version = '1.0'
        for pipe in published_pipelines:
            if name == pipe:
                pipeline_old = pipe
                version = str(np.float(pipeline_old.version) + 1.0)

        published_pipeline = self.pipeline.publish_pipeline(name, description, version)
    
        return published_pipeline
        
    def list_child_run_metrics(self, pipeline_run: PipelineRun):
        """List all metrics logged in child runs of a pipeline run.
        
        Parameters
        ----------
        pipeline_run : PipelineRun
            Azure ML PipelineRun instance.
        """
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
        for model in Model.list(ws):
            print(model.name, 'version:', model.version)
            for tag_name in model.tags:
                tag = model.tags[tag_name]
                print ('\t',tag_name, ':', tag)
            for prop_name in model.properties:
                prop = model.properties[prop_name]
                print ('\t',prop_name, ':', prop)
            print('\n')
