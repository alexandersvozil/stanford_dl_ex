function [out] = sigmoidDeriv(in)
	out = sigmoid(in) .* (1-sigmoid(in));
end
