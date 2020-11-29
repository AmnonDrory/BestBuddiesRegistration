Code for the paper "Best Buddies Registration for Point Clouds" by Amnon Drory, Tal Shomer, Shai Avidan and Raja Giryes from Tel-Aviv University. 

This code runs an experiment using the Horse point cloud (taken from https://www.cc.gatech.edu/projects/large_models/)

Dependencies
------------
A list of dependencies appears in dependencies/requirements.txt

A conda environment is also supplied. You can use it by running:

	conda env create -f dependencies/BBR_conda_env.yml
	
	conda activate BBR
	
	
Running
-------
To run the experiment:

	cd code
	
	python -m bb_pc.experiments.Horse
	
