%% A Benchmark Case for Statistic Process Monitoring - Cranfield Multiphase Flow Facility
% This package contains data sets collected in the Cranfield Multiphase
% Flow Facility aiming to serve as a benchmark case for statistic process
% monitoring. Details of the benchmark case are presented in [1]. Some
% examples of Canonical Variate Analysis (CVA) on these data sets are
% presented as follows. 
%% Reference
% [1] C. Ruiz-Cárcel, Y. Cao, D. Mba, L.Lao and R. T. Samuel, Statistical
% Process Monitoring of a Multiphase Flow Facility, _Control Engineering
% Practice_ , V. 42, PP. 74–88, 2015, <http://www.sciencedirect.com/science/article/pii/S0967066115000866#>  

%% Section 1 Parameters
%
vIndex = 1:23; % measurement index
alpha = 0.99;  % confidence level
n = 25;        % retained state dimension
p = 15;        % length of past observation
f = 15;        % length of future observation

%% Section 2 Training with normal data sets 2 and 3
% This block loads the available training data sets and selects
% the two data sets used to train the algorithm. The measurements included
% in both data sets are selected according to vIndex
load Training
X1 = T2(:,vIndex);
X2 = T3(:,vIndex);

%% Section 3 Fault detection for Faulty Case 3
% This block loads the data set used for the first monitoring example,
% including the same measurements as the training data sets (vIndex)
load FaultyCase3
XT = Set3_1(:,vIndex);

%% Section 4 Fault detection result Figure 7(a) and (b) in the paper
cvatutor(alpha, n,p,f,X1,X2,XT);

%% Section 5 Fault detection for Faulty Case 5
% This block loads the data set used for the second monitoring example,
% including the same measurements as the training data sets (vIndex)
load FaultyCase5
XT2 = Set5_2(:,vIndex);

%% Section 6 Fault detection result Figure 9(a) and (b) in the paper
cvatutor(alpha, n,p,f,X1,X2,XT2);