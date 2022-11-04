import pytest

from dsa.compute import AzureCompute


def test_compute():
    ac = AzureCompute(
        env_yml='environment.yml',
        compute_name='azdatascientist',
    )

    ac.list_environments()

    ac.delete_compute()
