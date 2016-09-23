function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

m = size(data,2);
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
s = cell(numHidden+1, 1);
s{1}.a_l = data;

for d = 1:numel(stack)
	curW = stack{d}.W; 
	curb = stack{d}.b;
	s{d+1}.z_l= bsxfun(@plus, curW * s{d}.a_l , curb);

	if(strcmp(ei.activation_fun,'logistic'))
		s{d+1}.a_l=sigmoid(s{d+1}.z_l);
	end

	if(strcmp(ei.activation_fun,'ReLU'))
		s{d+1}.a_l= max(s{d+1}.z_l,0);
	end

end


z_L = s{numel(stack)+1}.z_l;
probK = bsxfun(@rdivide,exp(z_L),sum(exp(z_L)));
a_L = probK;
s{numel(stack)+1}.a_l = a_L;

%% return here if only predictions desired.
if po
	cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
	grad = [];  
	pred_prob = a_L;
	return;
end;

%% compute cost
%squash the labels into a format where we can subtract it from the output layer
y_indic_tmp = eye(size(probK,1));
y_indic = y_indic_tmp(:,labels);
cost = - sum(sum(y_indic.*log(probK)))/m;


%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
deltaStack = cell(numHidden+1, 1);
delta_n_L= (a_L - y_indic);


deltaStack{numHidden+1}= delta_n_L;
gradStack{numHidden+1}.W = deltaStack{numHidden+1} *s{numHidden+1}.a_l'/m;
gradStack{numHidden+1}.b =sum(deltaStack{numHidden+1},2)/m;

for l = numHidden:-1:1
	
	if(strcmp(ei.activation_fun,'logistic'))
		deltaStack{l} = (stack{l+1}.W' * deltaStack{l+1}) .*...
		sigmoidDeriv(s{l+1}.z_l);
	end

	if(strcmp(ei.activation_fun,'ReLU'))
		deltaStack{l} = (stack{l+1}.W' * deltaStack{l+1}) .*...
		(max(s{l+1}.z_l,0)./s{l+1}.z_l);
	end
	gradStack{l}.W = deltaStack{l}*s{l}.a_l'/m;
	gradStack{l}.b = sum(deltaStack{l},2)/m;
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
costdecay = 0;
for l=1:numHidden+1
	gradStack{l}.W = gradStack{l}.W + (ei.lambda/m) * gradStack{l}.W;
	costdecay	   =  costdecay +  sum(sum(stack{l}.W.^2));
end
cost = cost + costdecay* (ei.lambda/(2*m));
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



