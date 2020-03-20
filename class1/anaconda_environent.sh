source ~/.bashrc
conda create -n dcon python=3.6 anaconda tensorflow keras opencv 
conda activate dcon
python -m ipykernel install --user --name dcon
jupyter notebook


