function Z = sampleGP(GP,DEBUG)
%function Z = sampleGP(GP)
% Generates a sample Z from a Gaussian process GP (see generateGP.m)

if nargin < 2, DEBUG = false; end

Z = GP.Value; A = find(isnan(Z)); n = length(A); % Identify hidden components
Z(A) = GP.Mu + chol(GP.Sigma)'*randn(n,1);
ldSig = log(det(GP.Sigma)); 
llZ = -0.5*(ldSig + mrdivide((Z(A)-GP.Mu)',GP.Sigma)*(Z(A)-GP.Mu) + n*log(2*pi));
llM = -0.5*(ldSig + n*log(2*pi));
%disp(['Log-likelihood of Z = ' num2str(llZ) ' (relative to Mu of ' num2str(llM) ')'])

if DEBUG
  showField(GP,Z,['Sample: L(Z) = ' num2str(llZ) ' (while L(\mu) = ' num2str(llM) ')'],GP.Mu);
end
