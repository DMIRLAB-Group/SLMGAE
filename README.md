# SLMGAE
##### Requirements
1. Python == 3.6
2. Tensorflow == 1.1.13
3. Numpy == 1.16.2
4. Scipy == 1.2

##### Repository Structure
+ SLMGAE/data/:
	+ ./List_Proteins_in_SL.txt: Contain all the genes involvd in SL pairs in our dataset.
	+ ./SL_Human_Approved.txt: Contain all the SL pairs, denoted as SynLethDB.
	+ ./biogrid_ppi_sparse.txt: Prtein-protein interactions data from BIOGRID dataset.
	+ ./Human_GOsim.txt: Gene ontology similarity(biological process).
	+ ./Human_GOsim_CC.txt£ºGene ontology similarity(cellular component).
+ SLMGAE/SLMGAE:
	+ ./inits.py: Initialize.
	+ ./layers.py: Implementation of each layers of our model.
	+ ./metrics.py: Evaluate functions.
	+ ./models.py: Implementation of our model.
	+ ./objective.py: Objective function of our model.
	+ ./train.py: Train our model.
	+ ./utils.py: Some utils(e.g., normalization).