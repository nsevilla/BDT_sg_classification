# BDT_sg_classification
In this directory you will find two scripts to be run in the ROOT environment. This has been tested in v5.18. NOTE THAT NEWER VERSIONS INTRODUCE CHANGES IN BDT PARAMETER NAMES! 

Using the data available at http://wwwae.ciemat.es/~sevilla/bdts, the workflow would be:
1) Train: root -l TMVAClassification_BDT.C
Note that you might have to change the directory or filename (look for train_dr9.root).
Also this script accepts as optional inputs the galaxy sample size to be used, star sample size and ntrees, nevmin, ncuts, maxdepth parameters
2) Evaluate: root -l TMVAClassification_Application.C
This produces a new ROOT tree in the 'newtree.root' file where the BDT classification will be dumped.
