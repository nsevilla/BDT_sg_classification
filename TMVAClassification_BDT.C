// @(#)root/tmva $Id: TMVAClassification.C 37399 2010-12-08 15:22:07Z evt $
/**********************************************************************************
 * Project   : TMVA - a ROOT-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Root Macro: TMVAClassification                                                 *
 *                                                                                *
 * This macro provides examples for the training and testing of the               *
 * TMVA classifiers.                                                              *
 *                                                                                *
 * As input data is used a toy-MC sample consisting of four Gaussian-distributed  *
 * and linearly correlated input variables.                                       *
 *                                                                                *
 * The methods to be used can be switched on and off by means of booleans, or     *
 * via the prompt command, for example:                                           *
 *                                                                                *
 *    root -l ./TMVAClassification.C\(\"Fisher,Likelihood\"\)                     *
 *                                                                                *
 * (note that the backslashes are mandatory)                                      *
 * If no method given, a default set of classifiers is used.                      *
 *                                                                                *
 * The output file "TMVA.root" can be analysed with the use of dedicated          *
 * macros (simply say: root -l <macro.C>), which can be conveniently              *
 * invoked through a GUI that will appear at the end of the run of this macro.    *
 *
 Launch the GUI via the command:                                                *
 *                                                                                *
 *    root -l ./TMVAGui.C                                                         *
 *                                                                                *
 **********************************************************************************/

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVAGui.C"

#if not defined(__CINT__) || defined(__MAKECINT__)
// needs to be included when makecint runs (ACLIC)
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#endif

void TMVAClassification_BDT( TString myMethodList = "" , Int_t ntrees = 2000, Int_t nevmin = 50, Int_t maxdepth = 15, Int_t ncuts = 200, Int_t ntrain = 30000, Int_t nbckg = 6000)
{
   // This loads the library
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // --- Boosted Decision Trees
   Use["BDT"]             = 0; // uses Adaptive Boost
   Use["BDTG"]            = 0; // uses Gradient Boost
   Use["BDTB"]            = 0; // uses Bagging
   Use["BDTD"]            = 1; // decorrelation + Adaptive Boost
   // 

   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVAClassification" << std::endl;

   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = TMVA::gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
            std::cout << std::endl;
            return;
         }
         Use[regMethod] = 1;
      }
   }

   // --------------------------------------------------------------------------------------------------

   // --- Here the preparation phase begins

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName( "" );
   //CHANGE HERE
   //outfileName = Form("tmva_training/results_BDT/TMVA_BDT_%d_%d.root",ntrain,nbckg);
   outfileName = Form("tmva_training/results_BDT_timing/TMVA_BDT_%d_%d_%d_%d.root",ntrees,nevmin,maxdepth,ncuts);
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Create the factory object. Later you can choose the methods
   // whose performance you'd like to investigate. The factory is 
   // the only TMVA object you have to interact with
   //
   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/
   //
   TString weightsBaseName("");
   //CHANGE HERE
   //weightsBaseName = Form("TMVAClassification_BDT_%d_%d",ntrain,nbckg);
   weightsBaseName = Form("TMVAClassification_BDT_%d_%d_%d_%d",ntrees,nevmin,maxdepth,ncuts);
   // The second argument is the output file for the training results
   // All TMVA output can be suppressed by removing the "!" (not) in
   // front of the "Silent" argument in the option string
   TMVA::Factory *factory = new TMVA::Factory( weightsBaseName, outputFile,
                                               "V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

   // Define the input variables that shall be used for the MVA training

    factory->AddVariable("petror50_r",'F');
    factory->AddVariable("petror90_r",'F');
    factory->AddVariable("lnlstar_r",'F');
    factory->AddVariable("lnlexp_r",'F');
    factory->AddVariable("lnldev_r",'F');
    factory->AddVariable("me1_r",'F');
    factory->AddVariable("me2_r",'F');
    factory->AddVariable("mrrcc_r",'F');

    factory->AddVariable("fibermag_u-fibermag_g",'F');
    factory->AddVariable("fibermag_g-fibermag_r",'F');
    factory->AddVariable("fibermag_r-fibermag_i",'F');
    factory->AddVariable("fibermag_i-fibermag_z",'F');
    factory->AddVariable("psfmag_u-psfmag_g",'F');
    factory->AddVariable("psfmag_g-psfmag_r",'F');
    factory->AddVariable("psfmag_r-psfmag_i",'F');
    factory->AddVariable("psfmag_i-psfmag_z",'F');
    factory->AddVariable("modelmag_u-modelmag_g",'F');
    factory->AddVariable("modelmag_g-modelmag_r",'F');
    factory->AddVariable("modelmag_r-modelmag_i",'F');
    factory->AddVariable("modelmag_i-modelmag_z",'F');
    factory->AddVariable("petromag_u-petromag_g",'F');
    factory->AddVariable("petromag_g-petromag_r",'F');
    factory->AddVariable("petromag_r-petromag_i",'F');
    factory->AddVariable("petromag_i-petromag_z",'F');

    factory->AddVariable("fibermag_r",'F');
    factory->AddVariable("psfmag_r",'F');
    factory->AddVariable("modelmag_r",'F');
    factory->AddVariable("petromag_r",'F');

   // You can add so-called "Spectator variables", which are not used in the MVA training,
   // but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
   // input variables, the response values of all trained MVAs, and the spectator variables
//    factory->AddSpectator( "spec1 := var1*2",  "Spectator 1", "units", 'F' );
//    factory->AddSpectator( "spec2 := var1*3",  "Spectator 2", "units", 'F' );

   // Read training and test data
   // (it is also possible to use ASCII format as input -> see TMVA Users Guide)
    TString fname = "train_dr9.root";  
    //TString fname = "/pc/desdsk01/des/sdss_nacho/train.root";

   if (gSystem->AccessPathName( fname )){  // file does not exist in local directory
     cout<<fname<<" NOT FOUND"<<endl;
     return; 
   }
   
   TFile *input = TFile::Open( fname );
   
   std::cout << "--- TMVAClassification       : Using input file: " << input->GetName() << std::endl;
   
   // --- Register the training and test trees

   TTree* inputTree = (TTree *) input->Get("To");
   //gROOT->cd();//this should fix the 'Failed filling branch' errors
   TCut signalCut = "specclass==2";
   TCut backgrCut = "specclass==1";
   factory->SetInputTrees(inputTree,signalCut,backgrCut);

   // global event weights per tree (see below for setting event-wise weights)
   //    Double_t signalWeight     = 1.0;
   //    Double_t backgroundWeight = 1.0;
   
   // Set individual event weights (the variables must exist in the original TTree)
   //    for signal    : factory->SetSignalWeightExpression    ("weight1*weight2");
   //    for background: factory->SetBackgroundWeightExpression("weight1*weight2");
   //    factory->SetBackgroundWeightExpression( "weight" );


   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = "petror50_r!=-9999&&me1_r!=-9999&&me2_r!=-9999&&petror90_r!=-9999&&modelmag_r<23"; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = "petror50_r!=-9999&&me1_r!=-9999&&me2_r!=-9999&&petror90_r!=-9999&&modelmag_r<23"; // for example: TCut mycutb = "abs(var1)<0.5";
   

   // Tell the factory how to use the training and testing events
   //
   // If no numbers of events are given, half of the events in the tree are used 
   // for training, and the other half for testing:
   TString training_string = Form("nTrain_Signal=%d:nTrain_Background=%d:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:V",ntrain,nbckg);
   factory->PrepareTrainingAndTestTree(mycuts, mycutb, training_string);
   
   // ---- Book MVA methods
   //  
   // Please lookup the various method configuration options in the corresponding cxx files, eg:
   // src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
   // it is possible to preset ranges in the option string in which the cut optimisation should be done:
   // "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable

   // Cut optimisation

   TString bookmethod_string = Form("!H:!V:NTrees=%d:nEventsMin=%d:MaxDepth=%d:nCuts=%d:BoostType=AdaBoost:SeparationType=GiniIndex:PruneMethod=NoPruning:VarTransform=Decorrelate",ntrees,nevmin,maxdepth,ncuts);

   // Boosted Decision Trees
   if (Use["BDTG"]) // Gradient Boost
      factory->BookMethod( TMVA::Types::kBDT, "BDTG", bookmethod_string);
   if (Use["BDT"])  // Adaptive Boost
     factory->BookMethod( TMVA::Types::kBDT, "BDT", bookmethod_string);
   if (Use["BDTB"]) // Bagging
      factory->BookMethod( TMVA::Types::kBDT, "BDTB", bookmethod_string);
   if (Use["BDTD"]) // Decorrelation + Adaptive Boost
      factory->BookMethod( TMVA::Types::kBDT, "BDTD", bookmethod_string);

   // For an example of the category classifier usage, see: TMVAClassificationCategory

   // --------------------------------------------------------------------------------------------------

   // ---- Now you can optimize the setting (configuration) of the MVAs using the set of training events

   // factory->OptimizeAllMethods("SigEffAt001","Scan");
   // factory->OptimizeAllMethods("ROCIntegral","GA");

   // --------------------------------------------------------------------------------------------------

   // ---- Now you can tell the factory to train, test, and evaluate the MVAs

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // ---- Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // ----- Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;

   delete factory;

   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVAGui( outfileName );
}
