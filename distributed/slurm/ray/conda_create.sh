module load Anaconda3/2019.03-gimkl-2018b
conda create -p /nesi/project/<id>/rayenv 
conda activate /nesi/project/<id>/rayenv
conda install --name ray pip
pip install --user ray
