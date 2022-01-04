%  Matlab Code-Library for Feature Selection
%  A collection of S-o-A feature selection methods
%  Version 5.1 October 2017
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk
%
%  Before using the Code-Library, please read the Release Agreement carefully.
%
%  Release Agreement:
%
%  - All technical papers, documents and reports which use the Code-Library will acknowledge the use of the library as follows: 
%    �The research in this paper use the Feature Selection Code Library (FSLib)� and a citation to:
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV17, 
% author={Giorgio Roffo and Simone Melzi and Umberto Castellani and Alessandro Vinciarelli}, 
% booktitle={2017 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach}, 
% year={2017}, 
% month={Oct}}
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV15, 
% author={G. Roffo and S. Melzi and M. Cristani}, 
% booktitle={2015 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Feature Selection}, 
% year={2015}, 
% pages={4202-4210}, 
% doi={10.1109/ICCV.2015.478}, 
% month={Dec}}
%  ------------------------------------------------------------------------

% Before using the toolbox compile the solution:
% make;

%% DEMO FILE
clear all
close all
clc;
fprintf('\nFEATURE SELECTION TOOLBOX v 5.0 2017 - For Matlab \n');
% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% Select a feature selection method from the list
listFS = {'ILFS','InfFS','ECFS', 'cfs', 'llcfs'};

[ methodID ] = readInput( listFS );
selection_method = listFS{methodID}; % Selected

% Load the data and select features for classification
%load fisheriris
%X = meas; clear meas
% Extract the Setosa class
%Y = nominal(ismember(species,'setosa')); clear species
X = csvread('D:\TIRTH\New folder\Sampled_D1.csv');
Y= X(:,220);
X(:,220) = [];
%X=X(:,[21:40])
% Randomly partitions observations into a training set and a test
% set using stratified holdout
%P = cvpartition(Y,'Holdout',0.20);

X_train = double( X );
Y_train = (double( Y )-1)*2-1; % labels: neg_class -1, pos_class +1

%X_test = double( X(P.test,:) );
%Y_test = (double( Y(P.test) )-1)*2-1; % labels: neg_class -1, pos_class +1
    
% number of features
%numF = size(X_train,354);


% feature Selection on training data
switch lower(selection_method)
    case 'ilfs'
        % Infinite Latent Feature Selection - ICCV 2017
        [ranking, weights, subset] = ILFS(X_train, Y_train , 4, 0 );
    
        
    case 'inffs'
        % Infinite Feature Selection 2015 updated 2016
        alpha = 0.5;    % default, it should be cross-validated.
        sup = 1;        % Supervised or Not
        [ranking, w] = infFS( X_train , Y_train, 0.5 , 1 , 0, 0, 'Z' );    
        
    case 'ecfs'
        % Features Selection via Eigenvector Centrality 2016
        alpha = 0.5; % default, it should be cross-validated.
        ranking = ECFS( X_train, Y_train, alpha )  ;
        
    case 'cfs'
        % BASELINE - Sort features according to pairwise correlations
        ranking = cfs(X_train);     
        
    case 'llcfs'   
        % Feature Selection and Kernel Learning for Local Learning-Based Clustering
        ranking = llcfs( X_train );
        
    otherwise
        disp('Unknown method.')
end

%k = 2; % select the first 2 features


disp('X_train size')
size(X_train)

disp('Y_train size')
size(Y_train)

%disp('X_test size')
%size(X_test)

%disp('Y_test size')
%size(Y_test)

ranking = transpose(ranking)
%fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f \n',selection_method,100*(1-err_rate),err_rate);



% MathWorks Licence
% Copyright (c) 2016-2017, Giorgio Roffo
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the University of Verona nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
