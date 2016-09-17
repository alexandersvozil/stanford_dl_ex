function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
fieldz = 'z_l';
valuez = [];
fielda = 'a_l';
valuea = [data];

%s - struct where all the intermediate results are saved
s = struct(fieldz,valuez, fielda, valuea);

for d = 1:numel(stack)
    curW = stack{d}.W; 
	curb = stack{d}.b;
	s(d+1).z_l= bsxfun(@plus, curW * s(d).a_l, curb);
	s(d+1).a_l=sigmoid(s(d+1).z_l);
end


%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
a_L = s(numel(stack)+1).a_l


%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



