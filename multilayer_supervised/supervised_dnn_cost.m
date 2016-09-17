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
m = size(data,2);
%% forward prop
%%% YOUR CODE HERE %%%
%fieldz = 'z_l';
%valuez = [];
%fielda = 'a_l';
%valuea = [data];

%s - struct where all the intermediate results are saved
%s = struct(fieldz,valuez, fielda, valuea);

s = cell(numHidden+1, 1);
s{1}.a_l = data;

for d = 1:numel(stack)
    curW = stack{d}.W; 
	curb = stack{d}.b;
	zwobias = curW * s{d}.a_l;
	s{d+1}.z_l= bsxfun(@plus, zwobias , curb);
	s{d+1}.a_l=sigmoid(s{d+1}.z_l);
end


a_L = s{numel(stack)+1}.a_l;
z_L = s{numel(stack)+1}.z_l;
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  pred_prob = a_L;
  return;
end;

%% compute cost


probK = bsxfun(@rdivide,exp(z_L),sum(exp(z_L)));
a_L = probK;

probKlog = log(probK);
%squash the labels into a format where we can subtract it from the output layer
y_indic = eye(size(probKlog,1))(:,labels);
cost = (-1) * sum(sum(y_indic.* probKlog));

%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
delta_n_L= - (y_indic - a_L);
%gradStack = cell(numHidden+1, 1);
%for l = 1:numHidden

%end
delta_n_l2	= (stack{2}.W' * delta_n_L) .* sigmoidDeriv(s{2}.z_l);

Delta_W1	= delta_n_l2*s{1}.a_l';
Delta_W2	= delta_n_L*s{2}.a_l';
Delta_b_1	= sum(delta_n_l2,2);
Delta_b_2	= sum(delta_n_L,2);

gradStack{1}.W = Delta_W1;
gradStack{1}.b = Delta_b_1;
gradStack{2}.W = Delta_W2;
gradStack{2}.b = Delta_b_2;


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



