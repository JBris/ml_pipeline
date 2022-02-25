"""
Machine Learning Pipeline Distributed

A library for facilitating distributed and parallel computing within the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
import os

##########################################################################################################
### Library  
##########################################################################################################

def init_dask():
    import dask.distributed as dd
    import dask_mpi as dm
    """Initialise Dask cluster files, and connect to cluster using a client."""
    # Initialise Dask cluster and store worker files in current work directory
    dm.initialize(local_directory = os.getcwd())
    client = dd.Client()
    return client

def close_dask(client):
    client.close()
