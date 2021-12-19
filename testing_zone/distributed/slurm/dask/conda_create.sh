module load Anaconda3/2019.03-gimkl-2018b
conda create -p /nesi/project/<ID>/daskenv \
  intel::mpi4py conda-forge::dask-mpi
