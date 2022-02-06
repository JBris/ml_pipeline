# conda create --prefix /nesi/project/ptec03219/ray_pycaret -y python=3.7
# conda activate /nesi/project/ptec03219/ray_pycaret
pip install --no-cache-dir optuna pycaret --user
pip install --no-cache-dir "ray[default]" "ray[tune]" --user
