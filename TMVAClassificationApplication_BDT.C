/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: TMVAClassificationApplication                                      *
 *                                                                                *
 * This macro provides a simple example on how to use the trained classifiers     *
 * within an analysis module                                                      *
 **********************************************************************************/

#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"

#include "TMVAGui.C"

#if not defined(__CINT__) || defined(__MAKECINT__)
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#endif

using namespace TMVA;

void TMVAClassificationApplication_BDT( TString myMethodList = "" , Int_t ntrees = 2000, Int_t nevmin = 50, Int_t maxdepth = 15, Int_t ncuts = 200, Int_t ntrain = 30000, Int_t nbckg = 6000) 
{   
#ifdef __CINT__
   gROOT->ProcessLine( ".O0" ); // turn off optimization in CINT
#endif

   //---------------------------------------------------------------

   // This loads the library
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // --- Cut optimisation
   Use["Cuts"]            = 0;
   Use["CutsD"]           = 0;
   Use["CutsPCA"]         = 0;
   Use["CutsGA"]          = 0;
   Use["CutsSA"]          = 0;
   // 
   // --- 1-dimensional likelihood ("naive Bayes estimator")
   Use["Likelihood"]      = 0;
   Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
   Use["LikelihoodPCA"]   = 0; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
   Use["LikelihoodKDE"]   = 0;
   Use["LikelihoodMIX"]   = 0;
   //
   // --- Mutidimensional likelihood and Nearest-Neighbour methods
   Use["PDERS"]           = 0;
   Use["PDERSD"]          = 0;
   Use["PDERSPCA"]        = 0;
   Use["PDEFoam"]         = 0;
   Use["PDEFoamBoost"]    = 0; // uses generalised MVA method boosting
   Use["KNN"]             = 0; // k-nearest neighbour method
   //
   // --- Linear Discriminant Analysis
   Use["LD"]              = 0; // Linear Discriminant identical to Fisher
   Use["Fisher"]          = 0;
   Use["FisherG"]         = 0;
   Use["BoostedFisher"]   = 0; // uses generalised MVA method boosting
   Use["HMatrix"]         = 0;
   //
   // --- Function Discriminant analysis
   Use["FDA_GA"]          = 0; // minimisation of user-defined function using Genetics Algorithm
   Use["FDA_SA"]          = 0;
   Use["FDA_MC"]          = 0;
   Use["FDA_MT"]          = 0;
   Use["FDA_GAMT"]        = 0;
   Use["FDA_MCMT"]        = 0;
   //
   // --- Neural Networks (all are feed-forward Multilayer Perceptrons)
   Use["MLP"]             = 0; // Recommended ANN
   Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
   Use["MLPBNN"]          = 0; // Recommended ANN with BFGS training method and bayesian regulator
   Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
   Use["TMlpANN"]         = 0; // ROOT's own ANN
   //
   // --- Support Vector Machine 
   Use["SVM"]             = 0;
   // 
   // --- Boosted Decision Trees
   Use["BDT"]             = 0; // uses Adaptive Boost
   Use["BDTG"]            = 0; // uses Gradient Boost
   Use["BDTB"]            = 0; // uses Bagging
   Use["BDTD"]            = 1; // decorrelation + Adaptive Boost
   // 
   // --- Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
   Use["RuleFit"]         = 0;
   // ---------------------------------------------------------------
   Use["Plugin"]          = 0;
   Use["Category"]        = 0;
   Use["SVM_Gauss"]       = 0;
   Use["SVM_Poly"]        = 0;
   Use["SVM_Lin"]         = 0;

   std::cout << std::endl;
   std::cout << "==> Start TMVAClassificationApplication" << std::endl;

   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod 
                      << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
               std::cout << it->first << " ";
            }
            std::cout << std::endl;
            return;
         }
         Use[regMethod] = 1;
      }
   }

   // --------------------------------------------------------------------------------------------------

   // --- Create the Reader object

   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );    

   // Create a set of variables and declare them to the reader
   // - the variable names MUST corresponds in name and type to those given in the weight file(s) used
   Float_t var[28];
   reader->AddVariable("petror50_r",&var[0]);
   reader->AddVariable("petror90_r",&var[1]);
   reader->AddVariable("lnlstar_r",&var[2]);
   reader->AddVariable("lnlexp_r",&var[3]);
   reader->AddVariable("lnldev_r",&var[4]);
   reader->AddVariable("me1_r",&var[5]);
   reader->AddVariable("me2_r",&var[6]);
   reader->AddVariable("mrrcc_r",&var[7]);
   reader->AddVariable("fibermag_u-fibermag_g",&var[8]);
   reader->AddVariable("fibermag_g-fibermag_r",&var[9]);
   reader->AddVariable("fibermag_r-fibermag_i",&var[10]);
   reader->AddVariable("fibermag_i-fibermag_z",&var[11]);
   reader->AddVariable("psfmag_u-psfmag_g",&var[12]);
   reader->AddVariable("psfmag_g-psfmag_r",&var[13]);
   reader->AddVariable("psfmag_r-psfmag_i",&var[14]);
   reader->AddVariable("psfmag_i-psfmag_z",&var[15]);
   reader->AddVariable("modelmag_u-modelmag_g",&var[16]);
   reader->AddVariable("modelmag_g-modelmag_r",&var[17]);
   reader->AddVariable("modelmag_r-modelmag_i",&var[18]);
   reader->AddVariable("modelmag_i-modelmag_z",&var[19]);
   reader->AddVariable("petromag_u-petromag_g",&var[20]);
   reader->AddVariable("petromag_g-petromag_r",&var[21]);
   reader->AddVariable("petromag_r-petromag_i",&var[22]);
   reader->AddVariable("petromag_i-petromag_z",&var[23]);
   reader->AddVariable("fibermag_r",&var[24]);
   reader->AddVariable("psfmag_r",&var[25]);
   reader->AddVariable("modelmag_r",&var[26]);
   reader->AddVariable("petromag_r",&var[27]);


//    // Spectator variables declared in the training have to be added to the reader, too
//    Float_t spec1,spec2;
//    reader->AddSpectator( "spec1 := var1*2",   &spec1 );
//    reader->AddSpectator( "spec2 := var1*3",   &spec2 );

//    Float_t Category_cat1, Category_cat2, Category_cat3;
//    if (Use["Category"]){
//       // Add artificial spectators for distinguishing categories
//       reader->AddSpectator( "Category_cat1 := var3<=0",             &Category_cat1 );
//       reader->AddSpectator( "Category_cat2 := (var3>0)&&(var4<0)",  &Category_cat2 );
//       reader->AddSpectator( "Category_cat3 := (var3>0)&&(var4>=0)", &Category_cat3 );
//    }

   // --- Book the MVA methods

   TString dir    = "weights/";
   //TString prefix = "TMVAClassification_BDT";

   // Book method(s)
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      if (it->second) {
         TString methodName = TString(it->first) + TString(" method");
         //TString weightfile = dir + prefix + TString("_") + TString(it->first) + TString(".weights.xml");
	 TString weightfile("");
	 //CHANGE HERE
	 weightfile = Form("TMVAClassification_BDT_%d_%d_%d_%d_BDTD.weights.xml",ntrees,nevmin,maxdepth,ncuts);
	 //weightfile = Form("TMVAClassification_BDT_%d_%d_BDTD.weights.xml",ntrain,nbckg);
         reader->BookMVA( methodName, dir + weightfile ); 
      }
   }
   
   // Book output histograms
   UInt_t nbin = 100;
   TH1F   *histLk(0), *histLkD(0), *histLkPCA(0), *histLkKDE(0), *histLkMIX(0), *histPD(0), *histPDD(0);
   TH1F   *histPDPCA(0), *histPDEFoam(0), *histPDEFoamErr(0), *histPDEFoamSig(0), *histKNN(0), *histHm(0);
   TH1F   *histFi(0), *histFiG(0), *histFiB(0), *histLD(0), *histNn(0),*histNnbfgs(0),*histNnbnn(0);
   TH1F   *histNnC(0), *histNnT(0), *histBdt(0), *histBdtG(0), *histBdtD(0), *histRf(0), *histSVMG(0);
   TH1F   *histSVMP(0), *histSVML(0), *histFDAMT(0), *histFDAGA(0), *histCat(0), *histPBdt(0);
   TH1F   *histNn_sta(0), *histNn_gal(0), *histBdt_sta_stdcut(0), *histBdt_gal_stdcut(0);
   TH2F   *histBdt_gal_stdcut_modelmag(0), *histBdt_sta_stdcut_modelmag(0), *histBdt_sta_modelmag(0), *histBdt_gal_modelmag(0);   
   TH2F   *histBdtD_sta_modelmag(0), *histBdtD_gal_modelmag(0);   
   TH2F   *histNn_gal_modelmag(0), *histNn_sta_modelmag(0);   

   if (Use["Likelihood"])    histLk      = new TH1F( "MVA_Likelihood",    "MVA_Likelihood",    nbin, -1, 1 );
   if (Use["LikelihoodD"])   histLkD     = new TH1F( "MVA_LikelihoodD",   "MVA_LikelihoodD",   nbin, -1, 0.9999 );
   if (Use["LikelihoodPCA"]) histLkPCA   = new TH1F( "MVA_LikelihoodPCA", "MVA_LikelihoodPCA", nbin, -1, 1 );
   if (Use["LikelihoodKDE"]) histLkKDE   = new TH1F( "MVA_LikelihoodKDE", "MVA_LikelihoodKDE", nbin,  -0.00001, 0.99999 );
   if (Use["LikelihoodMIX"]) histLkMIX   = new TH1F( "MVA_LikelihoodMIX", "MVA_LikelihoodMIX", nbin,  0, 1 );
   if (Use["PDERS"])         histPD      = new TH1F( "MVA_PDERS",         "MVA_PDERS",         nbin,  0, 1 );
   if (Use["PDERSD"])        histPDD     = new TH1F( "MVA_PDERSD",        "MVA_PDERSD",        nbin,  0, 1 );
   if (Use["PDERSPCA"])      histPDPCA   = new TH1F( "MVA_PDERSPCA",      "MVA_PDERSPCA",      nbin,  0, 1 );
   if (Use["KNN"])           histKNN     = new TH1F( "MVA_KNN",           "MVA_KNN",           nbin,  0, 1 );
   if (Use["HMatrix"])       histHm      = new TH1F( "MVA_HMatrix",       "MVA_HMatrix",       nbin, -0.95, 1.55 );
   if (Use["Fisher"])        histFi      = new TH1F( "MVA_Fisher",        "MVA_Fisher",        nbin, -4, 4 );
   if (Use["FisherG"])       histFiG     = new TH1F( "MVA_FisherG",       "MVA_FisherG",       nbin, -1, 1 );
   if (Use["BoostedFisher"]) histFiB     = new TH1F( "MVA_BoostedFisher", "MVA_BoostedFisher", nbin, -2, 2 );
   if (Use["LD"])            histLD      = new TH1F( "MVA_LD",            "MVA_LD",            nbin, -2, 2 );
   if (Use["MLP"])           histNn      = new TH1F( "MVA_MLP",           "MVA_MLP",           nbin, -1.25, 1.5 );
   if (Use["MLP"])           histNn_sta  = new TH1F( "MVA_MLP_sta",           "MVA_MLP_sta",           nbin, -1.25, 1.5 );
   if (Use["MLP"])           histNn_gal  = new TH1F( "MVA_MLP_gal",           "MVA_MLP_gal",           nbin, -1.25, 1.5 );
   if (Use["MLP"])           histNn_sta_modelmag  = new TH2F( "MVA_MLP_sta_modelmag",           "MVA_MLP_sta_modelmag",nbin, 13, 22, nbin, -1.25, 1.5 );
   if (Use["MLP"])           histNn_gal_modelmag  = new TH2F( "MVA_MLP_gal_modelmag",           "MVA_MLP_gal_modelmag",nbin, 13, 22, nbin, -1.25, 1.5 );
   if (Use["MLPBFGS"])       histNnbfgs  = new TH1F( "MVA_MLPBFGS",       "MVA_MLPBFGS",       nbin, -1.25, 1.5 );
   if (Use["MLPBNN"])        histNnbnn   = new TH1F( "MVA_MLPBNN",        "MVA_MLPBNN",        nbin, -1.25, 1.5 );
   if (Use["CFMlpANN"])      histNnC     = new TH1F( "MVA_CFMlpANN",      "MVA_CFMlpANN",      nbin,  0, 1 );
   if (Use["TMlpANN"])       histNnT     = new TH1F( "MVA_TMlpANN",       "MVA_TMlpANN",       nbin, -1.3, 1.3 );
   if (Use["BDT"])           histBdt     = new TH1F( "MVA_BDT",           "MVA_BDT",           nbin, -0.8, 0.8 );
   if (Use["BDT"])           histBdt_sta = new TH1F( "MVA_BDT_sta",       "MVA_BDT_sta",       nbin, -0.8, 0.8 );
   if (Use["BDT"])           histBdt_sta_stdcut = new TH1F( "MVA_BDT_sta_stdcut",       "MVA_BDT_sta_stdcut",       nbin, -0.8, 3.0 );
   if (Use["BDT"])           histBdt_sta_modelmag = new TH2F( "MVA_BDT_sta_modelmag",       "MVA_BDT_sta_modelmag",nbin, 13, 22, nbin, -0.8, 0.8);
   if (Use["BDT"])           histBdt_sta_stdcut_modelmag = new TH2F( "MVA_BDT_sta_stdcut_modelmag",       "MVA_BDT_sta_stdcut_modelmag",nbin, 13, 22, nbin, -0.8, 3.0 );
   if (Use["BDT"])           histBdt_gal = new TH1F( "MVA_BDT_gal",       "MVA_BDT_gal",       nbin, -0.8, 0.8 );
   if (Use["BDT"])           histBdt_gal_stdcut = new TH1F( "MVA_BDT_gal_stdcut",       "MVA_BDT_gal_stdcut",       nbin, -0.8, 3.0 );
   if (Use["BDT"])           histBdt_gal_modelmag = new TH2F( "MVA_BDT_gal_modelmag",       "MVA_BDT_gal_modelmag",nbin, 13, 22, nbin, -0.8, 0.8 );
   if (Use["BDT"])           histBdt_gal_stdcut_modelmag = new TH2F( "MVA_BDT_gal_stdcut_modelmag",       "MVA_BDT_gal_stdcut_modelmag",nbin, 13, 22, nbin, -0.8, 3.0 );
   if (Use["BDTD"])          histBdtD    = new TH1F( "MVA_BDTD",          "MVA_BDTD",          nbin, -0.8, 0.8 );
   if (Use["BDTD"])          histBdtD_sta = new TH1F( "MVA_BDTD_sta",       "MVA_BDTD_sta",       nbin, -0.8, 0.8 );
   if (Use["BDTD"])          histBdtD_sta_modelmag = new TH2F( "MVA_BDTD_sta_modelmag",       "MVA_BDTD_sta_modelmag",nbin, 13, 22, nbin, -0.8, 0.8);
   if (Use["BDTD"])          histBdtD_gal = new TH1F( "MVA_BDTD_gal",       "MVA_BDTD_gal",       nbin, -0.8, 0.8 );
   if (Use["BDTD"])          histBdtD_gal_modelmag = new TH2F( "MVA_BDTD_gal_modelmag",       "MVA_BDTD_gal_modelmag",nbin, 13, 22, nbin, -0.8, 0.8 );
   if (Use["BDTG"])          histBdtG    = new TH1F( "MVA_BDTG",          "MVA_BDTG",          nbin, -1.0, 1.0 );
   if (Use["RuleFit"])       histRf      = new TH1F( "MVA_RuleFit",       "MVA_RuleFit",       nbin, -2.0, 2.0 );
   if (Use["SVM_Gauss"])     histSVMG    = new TH1F( "MVA_SVM_Gauss",     "MVA_SVM_Gauss",     nbin,  0.0, 1.0 );
   if (Use["SVM_Poly"])      histSVMP    = new TH1F( "MVA_SVM_Poly",      "MVA_SVM_Poly",      nbin,  0.0, 1.0 );
   if (Use["SVM_Lin"])       histSVML    = new TH1F( "MVA_SVM_Lin",       "MVA_SVM_Lin",       nbin,  0.0, 1.0 );
   if (Use["FDA_MT"])        histFDAMT   = new TH1F( "MVA_FDA_MT",        "MVA_FDA_MT",        nbin, -2.0, 3.0 );
   if (Use["FDA_GA"])        histFDAGA   = new TH1F( "MVA_FDA_GA",        "MVA_FDA_GA",        nbin, -2.0, 3.0 );
   if (Use["Category"])      histCat     = new TH1F( "MVA_Category",      "MVA_Category",      nbin, -2., 2. );
   if (Use["Plugin"])        histPBdt    = new TH1F( "MVA_PBDT",          "MVA_BDT",           nbin, -0.8, 0.8 );

   // PDEFoam also returns per-event error, fill in histogram, and also fill significance
   if (Use["PDEFoam"]) {
      histPDEFoam    = new TH1F( "MVA_PDEFoam",       "MVA_PDEFoam",              nbin,  0, 1 );
      histPDEFoamErr = new TH1F( "MVA_PDEFoamErr",    "MVA_PDEFoam error",        nbin,  0, 1 );
      histPDEFoamSig = new TH1F( "MVA_PDEFoamSig",    "MVA_PDEFoam significance", nbin,  0, 10 );
   }

   // Book example histogram for probability (the other methods are done similarly)
   TH1F *probHistFi(0), *rarityHistFi(0);
   if (Use["Fisher"]) {
      probHistFi   = new TH1F( "MVA_Fisher_Proba",  "MVA_Fisher_Proba",  nbin, 0, 1 );
      rarityHistFi = new TH1F( "MVA_Fisher_Rarity", "MVA_Fisher_Rarity", nbin, 0, 1 );
   }

   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.
   //   
   TFile *input(0);
      
   TString fname = "eval_dr9.root";  

   if (!gSystem->AccessPathName( fname )) 
      input = TFile::Open( fname ); // check if file in local directory exists
   else    
      input = TFile::Open( "http://root.cern.ch/files/tmva_class_example.root" ); // if not: download from ROOT server
     
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVAClassificationApp    : Using input file: " << input->GetName() << std::endl;
   
   // --- Event loop

   // Prepare the event tree
   // - here the variable names have to corresponds to your tree
   // - you can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   //
   std::cout << "--- Select signal sample" << std::endl;
   TTree* inputTree = (TTree *) input->Get("To");
   gROOT->cd();//this should fix the 'Failed filling branch' errors
   Double_t ra,dec,psfmag_u,psfmag_g,psfmag_r,psfmag_i,psfmag_z,modelmag_u,modelmag_g,modelmag_r,modelmag_i,modelmag_z,petromag_u,petromag_g,petromag_r,petromag_i,petromag_z,fibermag_u,fibermag_g,fibermag_r,fibermag_i,fibermag_z,petrorad_r,petror50_r,petror90_r,lnlstar_r,lnlexp_r,lnldev_r,me1_r,me2_r,mrrcc_r;
   Float_t bdtvar,bdtdvar,mlpvar;
   Int_t type_r,type,specclass;
   inputTree->SetBranchAddress( "ra", &ra );
   inputTree->SetBranchAddress( "dec", &dec );
   inputTree->SetBranchAddress( "psfmag_u", &psfmag_u );
   inputTree->SetBranchAddress( "psfmag_g", &psfmag_g );
   inputTree->SetBranchAddress( "psfmag_r", &psfmag_r );
   inputTree->SetBranchAddress( "psfmag_i", &psfmag_i );
   inputTree->SetBranchAddress( "psfmag_z", &psfmag_z );
   inputTree->SetBranchAddress( "modelmag_u", &modelmag_u );
   inputTree->SetBranchAddress( "modelmag_g", &modelmag_g );
   inputTree->SetBranchAddress( "modelmag_r", &modelmag_r );
   inputTree->SetBranchAddress( "modelmag_i", &modelmag_i );
   inputTree->SetBranchAddress( "modelmag_z", &modelmag_z );
   inputTree->SetBranchAddress( "petromag_u", &petromag_u );
   inputTree->SetBranchAddress( "petromag_g", &petromag_g );
   inputTree->SetBranchAddress( "petromag_r", &petromag_r );
   inputTree->SetBranchAddress( "petromag_i", &petromag_i );
   inputTree->SetBranchAddress( "petromag_z", &petromag_z );
   inputTree->SetBranchAddress( "fibermag_u", &fibermag_u );
   inputTree->SetBranchAddress( "fibermag_g", &fibermag_g );
   inputTree->SetBranchAddress( "fibermag_r", &fibermag_r );
   inputTree->SetBranchAddress( "fibermag_i", &fibermag_i );
   inputTree->SetBranchAddress( "fibermag_z", &fibermag_z );
   inputTree->SetBranchAddress( "petrorad_r", &petrorad_r );
   inputTree->SetBranchAddress( "petror50_r", &petror50_r );
   inputTree->SetBranchAddress( "petror90_r", &petror90_r );
   inputTree->SetBranchAddress( "lnlstar_r", &lnlstar_r );
   inputTree->SetBranchAddress( "lnlexp_r", &lnlexp_r );
   inputTree->SetBranchAddress( "lnldev_r", &lnldev_r );
   inputTree->SetBranchAddress( "me1_r", &me1_r );
   inputTree->SetBranchAddress( "me2_r", &me2_r );
   inputTree->SetBranchAddress( "mrrcc_r", &mrrcc_r );
   inputTree->SetBranchAddress( "type_r", &type_r );
   inputTree->SetBranchAddress( "type", &type );
   inputTree->SetBranchAddress( "specclass", &specclass );
   
   //TString prefix("/home/sevilla/SCRATCH/");
   //TString outfilename("");   
   //outfilename = Form("tmp_BDT_%d_%d_%d_%d.root",ntrees,nevmin,maxdepth,ncuts);
   //TFile *outfile  = new TFile(prefix+outfilename,"RECREATE" ); // crea fichero

   //TTree *outputTree = inputTree->CloneTree(); // crea nuevo tree clonado del inputTree
   //TBranch* bdtd = outputTree->Branch("bdtdvar",&bdtdvar,"bdtdvar/F"); // añade nueva rama al tree
   TBranch* bdtd = inputTree->Branch("bdtdvar",&bdtdvar,"bdtdvar/F"); // añade nueva rama al tree

   // Efficiency calculator for cut method
   Int_t    nSelCuts = 0;
   Double_t effS     = 0.7;

   std::vector<Float_t> vecVar(9); // vector for EvaluateMVA tests

   std::cout << "--- Processing: " << inputTree->GetEntries() << " events" << std::endl;
   TStopwatch sw;
   sw.Start();
   Int_t ngal[9],ngal_sel[9],nsta_sel[9],ngal_sel_bdt[9],nsta_sel_bdt[9],ngal_sel_bdtd[9],nsta_sel_bdtd[9],ngal_sel_nn[9],nsta_sel_nn[9];
   Float_t mag[9],eff_std[9],eff_bdt[9],eff_bdtd[9],eff_nn[9],imp_std[9],imp_bdt[9],imp_bdtd[9],imp_nn[9];
   Int_t magbin;
   //Float_t sep_threshold[7] = {1.25,1.0,0.65,0.45,0.25,0.15,0.145};
   Float_t sep_threshold[9] = {0.145,0.145,0.145,0.145,0.145,0.145,0.145,0.145,0.145};

   for(Int_t m=0;m<9;m++) {
     ngal[m] = 0; ngal_sel[m] = 0; nsta_sel[m] = 0; ngal_sel_bdt[m] = 0; nsta_sel_bdt[m] = 0; ngal_sel_nn[m] = 0; nsta_sel_nn[m] = 0; ngal_sel_bdtd[m] = 0; nsta_sel_bdtd[m] = 0;
     mag[m] = 14.5+m;
   }

   for (Long64_t ievt=0; ievt<inputTree->GetEntries();ievt++) {
   //for (Long64_t ievt=0; ievt<10000;ievt++) {

     if (ievt%1000 == 0) 
       std::cout << "--- ... Processing event: " << ievt << std::endl;

      inputTree->GetEntry(ievt);
      
      var[0] = petror50_r;
      var[1] = petror90_r;
      var[2] = lnlstar_r;
      var[3] = lnlexp_r;
      var[4] = lnldev_r;
      var[5] = me1_r;
      var[6] = me2_r;
      var[7] = mrrcc_r;
      var[8] = fibermag_u-fibermag_g;
      var[9] = fibermag_g-fibermag_r;
      var[10] = fibermag_r-fibermag_i;
      var[11] = fibermag_i-fibermag_z;
      var[12] = psfmag_u-psfmag_g;
      var[13] = psfmag_g-psfmag_r;
      var[14] = psfmag_r-psfmag_i;
      var[15] = psfmag_i-psfmag_z;
      var[16] = modelmag_u-modelmag_g;
      var[17] = modelmag_g-modelmag_r;
      var[18] = modelmag_r-modelmag_i;
      var[19] = modelmag_i-modelmag_z;
      var[20] = petromag_u-petromag_g;
      var[21] = petromag_g-petromag_r;
      var[22] = petromag_r-petromag_i;
      var[23] = petromag_i-petromag_z;
      var[24] = fibermag_r;
      var[25] = psfmag_r;
      var[26] = modelmag_r;
      var[27] = petromag_r;

      // --- Evaluate efficiency and purity
      if(modelmag_r<=14.0||modelmag_r>=23.0) continue;
      magbin = int(modelmag_r-14.0);
      if(specclass==2) ngal[magbin]++;
      if(var[25]-var[26]>sep_threshold[magbin]){
        if(specclass==1){
          nsta_sel[magbin]++;
        }
        else if(specclass==2){
          ngal_sel[magbin]++;
        }
      }
      if(Use["BDT"          ]){
        bdtvar = reader->EvaluateMVA("BDT method");
        if(bdtvar>0.05){
          if(specclass==1){
            nsta_sel_bdt[magbin]++;
          }
          else if(specclass==2){
            ngal_sel_bdt[magbin]++;
          }
        }
        //bdt->Fill();
      }
      if(Use["BDTD"          ]){
        bdtdvar = reader->EvaluateMVA("BDTD method");
        if(bdtdvar>0.05){
          if(specclass==1){
            nsta_sel_bdtd[magbin]++;
          }
          else if(specclass==2){
            ngal_sel_bdtd[magbin]++;
          }
        }
        bdtd->Fill();
      }

      // --- Return the MVA outputs and fill into histograms

      if (Use["Cuts"]) {
         // Cuts is a special case: give the desired signal efficienciy
         Bool_t passed = reader->EvaluateMVA( "Cuts method", effS );
         if (passed) nSelCuts++;
      }

      if (Use["Likelihood"   ])   histLk     ->Fill( reader->EvaluateMVA( "Likelihood method"    ) );
      if (Use["LikelihoodD"  ])   histLkD    ->Fill( reader->EvaluateMVA( "LikelihoodD method"   ) );
      if (Use["LikelihoodPCA"])   histLkPCA  ->Fill( reader->EvaluateMVA( "LikelihoodPCA method" ) );
      if (Use["LikelihoodKDE"])   histLkKDE  ->Fill( reader->EvaluateMVA( "LikelihoodKDE method" ) );
      if (Use["LikelihoodMIX"])   histLkMIX  ->Fill( reader->EvaluateMVA( "LikelihoodMIX method" ) );
      if (Use["PDERS"        ])   histPD     ->Fill( reader->EvaluateMVA( "PDERS method"         ) );
      if (Use["PDERSD"       ])   histPDD    ->Fill( reader->EvaluateMVA( "PDERSD method"        ) );
      if (Use["PDERSPCA"     ])   histPDPCA  ->Fill( reader->EvaluateMVA( "PDERSPCA method"      ) );
      if (Use["KNN"          ])   histKNN    ->Fill( reader->EvaluateMVA( "KNN method"           ) );
      if (Use["HMatrix"      ])   histHm     ->Fill( reader->EvaluateMVA( "HMatrix method"       ) );
      if (Use["Fisher"       ])   histFi     ->Fill( reader->EvaluateMVA( "Fisher method"        ) );
      if (Use["FisherG"      ])   histFiG    ->Fill( reader->EvaluateMVA( "FisherG method"       ) );
      if (Use["BoostedFisher"])   histFiB    ->Fill( reader->EvaluateMVA( "BoostedFisher method" ) );
      if (Use["LD"           ])   histLD     ->Fill( reader->EvaluateMVA( "LD method"            ) );
      if (Use["MLP"          ])   histNn     ->Fill( reader->EvaluateMVA( "MLP method"           ) );
      if (Use["MLPBFGS"      ])   histNnbfgs ->Fill( reader->EvaluateMVA( "MLPBFGS method"       ) );
      if (Use["MLPBNN"       ])   histNnbnn  ->Fill( reader->EvaluateMVA( "MLPBNN method"        ) );
      if (Use["CFMlpANN"     ])   histNnC    ->Fill( reader->EvaluateMVA( "CFMlpANN method"      ) );
      if (Use["TMlpANN"      ])   histNnT    ->Fill( reader->EvaluateMVA( "TMlpANN method"       ) );
      if (Use["BDT"          ])   histBdt    ->Fill( reader->EvaluateMVA( "BDT method"           ) );
      if (Use["BDTD"         ])   histBdtD   ->Fill( reader->EvaluateMVA( "BDTD method"          ) );
      if (Use["BDTG"         ])   histBdtG   ->Fill( reader->EvaluateMVA( "BDTG method"          ) );
      if (Use["RuleFit"      ])   histRf     ->Fill( reader->EvaluateMVA( "RuleFit method"       ) );
      if (Use["SVM_Gauss"    ])   histSVMG   ->Fill( reader->EvaluateMVA( "SVM_Gauss method"     ) );
      if (Use["SVM_Poly"     ])   histSVMP   ->Fill( reader->EvaluateMVA( "SVM_Poly method"      ) );
      if (Use["SVM_Lin"      ])   histSVML   ->Fill( reader->EvaluateMVA( "SVM_Lin method"       ) );
      if (Use["FDA_MT"       ])   histFDAMT  ->Fill( reader->EvaluateMVA( "FDA_MT method"        ) );
      if (Use["FDA_GA"       ])   histFDAGA  ->Fill( reader->EvaluateMVA( "FDA_GA method"        ) );
      if (Use["Category"     ])   histCat    ->Fill( reader->EvaluateMVA( "Category method"      ) );
      if (Use["Plugin"       ])   histPBdt   ->Fill( reader->EvaluateMVA( "P_BDT method"         ) );
      if(specclass==1){
        if(Use["BDT"         ]){
          histBdt_sta->Fill( reader->EvaluateMVA( "BDT method"           ) );
          histBdt_sta_stdcut->Fill( var[25]-var[26] );
          histBdt_sta_modelmag->Fill( modelmag_r, reader->EvaluateMVA( "BDT method"           ) );
          histBdt_sta_stdcut_modelmag->Fill( modelmag_r,  var[25]-var[26]);
        }
        if(Use["BDTD"         ]){
          histBdtD_sta->Fill( reader->EvaluateMVA( "BDTD method"           ) );
          histBdtD_sta_modelmag->Fill( modelmag_r, reader->EvaluateMVA( "BDTD method"           ) );
        }
        if(Use["MLP"         ]){
          histNn_sta->Fill( reader->EvaluateMVA( "MLP method"           ) );
          histNn_sta_modelmag->Fill( modelmag_r, reader->EvaluateMVA( "MLP method"           ) );
        }
      }
      else if(specclass==2){
        if(Use["BDT"         ]){
          histBdt_gal->Fill( reader->EvaluateMVA( "BDT method"           ) );
          histBdt_gal_modelmag->Fill( modelmag_r, reader->EvaluateMVA( "BDT method"           ) );
        }
        if(Use["BDTD"         ]){
          histBdtD_gal->Fill( reader->EvaluateMVA( "BDTD method"           ) );
          histBdtD_gal_modelmag->Fill( modelmag_r, reader->EvaluateMVA( "BDTD method"           ) );
        }
        if(Use["MLP"         ]){
          histNn_gal->Fill( reader->EvaluateMVA( "MLP method"           ) );
          histNn_gal_modelmag->Fill( modelmag_r, reader->EvaluateMVA( "MLP method"           ) );
        }
      }      

   }

   inputTree->SetBranchStatus("*",0);
   inputTree->SetBranchStatus("modelmag_r",1);
   inputTree->SetBranchStatus("psfmag_r",1);
   inputTree->SetBranchStatus("specclass",1);
   inputTree->SetBranchStatus("bdtdvar",1);

   TString newfilename("");
   TString newprefix("./");
   newfilename = Form("newtree_BDT_%d_%d_%d_%d.root",ntrees,nevmin,maxdepth,ncuts);
   TFile *newfile = new TFile(newprefix+newfilename,"RECREATE");
   TTree *newtree = inputTree->CloneTree(0);
   newtree->CopyEntries(inputTree);
   newfile->Write();
   newfile->Close();

   for(Int_t m=0;m<9;m++){
     std::cout << "Magnitude "<< 14+m <<"-"<< 14+m+1 <<endl;
     eff_std[m] = (float(ngal_sel[m])/float(ngal[m]))*100;
     imp_std[m] = (float(nsta_sel[m])/float(ngal_sel[m]+nsta_sel[m]))*100;
     std::cout << "  STANDARD "<< endl; 
     std::cout << "  Selected galaxies = "<< float(ngal_sel[m])<<endl; 
     std::cout << "  Total real galaxies = "<< float(ngal[m])<<endl; 
     std::cout << "  Selected stars = "<< float(nsta_sel[m])<<endl; 
     std::cout << "  Total galaxies = "<< float(ngal_sel[m])+float(nsta_sel[m])<<endl; 
     std::cout << "  Efficiency (std. cut) = "<< eff_std[m] << "%"<<endl; 
     std::cout << "  Impurity (std.cut )= "<< imp_std[m] << "%"<<endl; 

     if(Use["BDT"         ]){
       eff_bdt[m] = (float(ngal_sel_bdt[m])/float(ngal[m]))*100;
       imp_bdt[m] = (float(nsta_sel_bdt[m])/float(ngal_sel_bdt[m]+nsta_sel_bdt[m]))*100;
       std::cout << "  BDT "<< endl; 
       std::cout << "  Selected galaxies = "<< float(ngal_sel_bdt[m])<<endl; 
       std::cout << "  Total real galaxies = "<< float(ngal[m])<<endl; 
       std::cout << "  Selected stars = "<< float(nsta_sel_bdt[m])<<endl; 
       std::cout << "  Total galaxies = "<< float(ngal_sel_bdt[m])+float(nsta_sel_bdt[m])<<endl; 
       std::cout << "  Efficiency (BDT cut) = "<< eff_bdt[m] << "%"<<endl; 
       std::cout << "  Impurity (BDT cut )= "<< imp_bdt[m] << "%"<<endl; 
     }
     if(Use["BDTD"         ]){
       eff_bdtd[m] = (float(ngal_sel_bdtd[m])/float(ngal[m]))*100;
       imp_bdtd[m] = (float(nsta_sel_bdtd[m])/float(ngal_sel_bdtd[m]+nsta_sel_bdtd[m]))*100;
       std::cout << "  BDTD "<< endl; 
       std::cout << "  Selected galaxies = "<< float(ngal_sel_bdtd[m])<<endl; 
       std::cout << "  Total real galaxies = "<< float(ngal[m])<<endl; 
       std::cout << "  Selected stars = "<< float(nsta_sel_bdtd[m])<<endl; 
       std::cout << "  Total galaxies = "<< float(ngal_sel_bdtd[m])+float(nsta_sel_bdtd[m])<<endl; 
       std::cout << "  Efficiency (BDTD cut) = "<< eff_bdtd[m] << "%"<<endl; 
       std::cout << "  Impurity (BDTD cut )= "<< imp_bdtd[m] << "%"<<endl; 
     }
   }


   // Get elapsed time
   sw.Stop();
   std::cout << "--- End of event loop: "; sw.Print();

   // Get efficiency for cuts classifier
   if (Use["CutsGA"]) std::cout << "--- Efficiency for CutsGA method: " << double(nSelCutsGA)/inputTree->GetEntries()
                                << " (for a required signal efficiency of " << effS << ")" << std::endl;

   if (Use["CutsGA"]) {

      // test: retrieve cuts for particular signal efficiency
      // CINT ignores dynamic_casts so we have to use a cuts-secific Reader function to acces the pointer  
      TMVA::MethodCuts* mcuts = reader->FindCutsMVA( "CutsGA method" ) ;

      if (mcuts) {      
         std::vector<Double_t> cutsMin;
         std::vector<Double_t> cutsMax;
         mcuts->GetCuts( 0.7, cutsMin, cutsMax );
         std::cout << "--- -------------------------------------------------------------" << std::endl;
         std::cout << "--- Retrieve cut values for signal efficiency of 0.7 from Reader" << std::endl;
         for (UInt_t ivar=0; ivar<cutsMin.size(); ivar++) {
            std::cout << "... Cut: " 
                      << cutsMin[ivar] 
                      << " < \"" 
                      << mcuts->GetInputVar(ivar)
                      << "\" <= " 
                      << cutsMax[ivar] << std::endl;
         }
         std::cout << "--- -------------------------------------------------------------" << std::endl;
      }
   }

   // --- Write histograms

   TString targetname("");
   //CHANGE HERE
   targetname = Form("TMVApp_BDT_%d_%d_%d_%d.root",ntrees,nevmin,maxdepth,ncuts);
   //targetname = Form("TMVApp_BDT_%d_%d.root",ntrain,nbckg);
   
   TFile *target  = new TFile( newprefix+targetname ,"RECREATE" );
   if (Use["Likelihood"   ])   histLk     ->Write();
   if (Use["LikelihoodD"  ])   histLkD    ->Write();
   if (Use["LikelihoodPCA"])   histLkPCA  ->Write();
   if (Use["LikelihoodKDE"])   histLkKDE  ->Write();
   if (Use["LikelihoodMIX"])   histLkMIX  ->Write();
   if (Use["PDERS"        ])   histPD     ->Write();
   if (Use["PDERSD"       ])   histPDD    ->Write();
   if (Use["PDERSPCA"     ])   histPDPCA  ->Write();
   if (Use["KNN"          ])   histKNN    ->Write();
   if (Use["HMatrix"      ])   histHm     ->Write();
   if (Use["Fisher"       ])   histFi     ->Write();
   if (Use["LD"           ])   histLD     ->Write();
   if (Use["MLP"          ])   histNn     ->Write();
   if (Use["MLP"          ])   histNn_sta     ->Write();
   if (Use["MLP"          ])   histNn_gal     ->Write();
   if (Use["MLP"          ])   histNn_sta_modelmag->Write();
   if (Use["MLP"          ])   histNn_gal_modelmag->Write();
   if (Use["MLPBFGS"      ])   histNnbfgs ->Write();
   if (Use["MLPBNN"       ])   histNnbnn  ->Write();
   if (Use["CFMlpANN"     ])   histNnC    ->Write();
   if (Use["TMlpANN"      ])   histNnT    ->Write();
   if (Use["BDT"          ])   histBdt    ->Write();
   if (Use["BDT"          ])   histBdt_sta->Write();
   if (Use["BDT"          ])   histBdt_sta_stdcut->Write();
   if (Use["BDT"          ])   histBdt_sta_modelmag->Write();
   if (Use["BDT"          ])   histBdt_sta_stdcut_modelmag->Write();
   if (Use["BDT"          ])   histBdt_gal->Write();
   if (Use["BDT"          ])   histBdt_gal_stdcut->Write();
   if (Use["BDT"          ])   histBdt_gal_modelmag->Write();
   if (Use["BDT"          ])   histBdt_gal_stdcut_modelmag->Write();
   if (Use["BDTD"         ])   histBdtD   ->Write();
   if (Use["BDTD"         ])   histBdtD_sta->Write();
   if (Use["BDTD"         ])   histBdtD_sta_modelmag->Write();
   if (Use["BDTD"         ])   histBdtD_gal->Write();
   if (Use["BDTD"         ])   histBdtD_gal_modelmag->Write();
   if (Use["BDTG"         ])   histBdtG   ->Write(); 
   if (Use["RuleFit"      ])   histRf     ->Write();
   if (Use["SVM_Gauss"    ])   histSVMG   ->Write();
   if (Use["SVM_Poly"     ])   histSVMP   ->Write();
   if (Use["SVM_Lin"      ])   histSVML   ->Write();
   if (Use["FDA_MT"       ])   histFDAMT  ->Write();
   if (Use["FDA_GA"       ])   histFDAGA  ->Write();
   if (Use["Category"     ])   histCat    ->Write();
   if (Use["Plugin"       ])   histPBdt   ->Write();


   // Write also error and significance histos
   if (Use["PDEFoam"]) { histPDEFoam->Write(); histPDEFoamErr->Write(); histPDEFoamSig->Write(); }

   // Write also probability hists
   if (Use["Fisher"]) { if (probHistFi != 0) probHistFi->Write(); if (rarityHistFi != 0) rarityHistFi->Write(); }
   target->Close();

   std::cout << "--- Created root file: \"TMVApp.root\" containing the MVA output histograms" << std::endl;
  
   delete reader;

   std::cout << "==> TMVAClassificationApplication is done!" << endl << std::endl;

}
