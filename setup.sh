module purge
module load GCCcore/13.3.0
module load Python/3.12.3
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# must be executed on a GPU node
# to have cuda12 
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
