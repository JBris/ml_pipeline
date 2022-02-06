conda create --prefix /nesi/project/ptec03219/tpot_dask -y python=3.7
conda activate /nesi/project/ptec03219/tpot_dask
conda install -y -c intel mpi4py
pip install --no-cache-dir dask dask-ml distributed dask-mpi --user 
pip install --no-cache-dir tpot --user 
# pip install mlflow --user 
