import os
import pytest
import pandas as pd

from dsa.pipeline import ExampleAzurePipeline


@pytest.fixture
def ap():
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
    return ap


class TestAzurePipeline:
    def test_init(self, ap):
        assert type(ap) == ExampleAzurePipeline
    
    def test_set_pipeline_run_config(self, ap):
        pipeline_run_config = ap.set_pipeline_run_config()
        assert pipeline_run_config.environment == ap.environment
        assert pipeline_run_config.target == ap.compute.name
    
    def test_define_pipeline_steps(self, ap):
        pipeline_steps = ap.define_pipeline_steps()
        assert pipeline_steps is not None
    
    def test_create_pipeline(self, ap):
        ap.create_pipeline()
    
    def test_run_pipeline(self, ap):
        with pytest.raises(ValueError):
            ap.run_pipeline()
        ap.create_pipeline()
        ap.run_pipeline()
        
    def publish_pipeline(self, ap):
        with pytest.raises(ValueError):
            ap.publish_pipeline()
        ap.create_pipeline()
        ap.publish_pipeline()
        
    def test_list_child_run_metrics(self, ap):
        ap.list_child_run_metrics()  # No pipeline run defined (warning to stdout)

        ap.create_pipeline()
        ap.run_pipeline()
        ap.list_child_run_metrics()  # Pipeline run is defined, get run metrics
    
    def test_list_registered_models(self, ap):
        ap.list_registered_models()
