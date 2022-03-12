function GP = generateGP(M,N,phi,DEBUG,mumix)
%function GP = generateGP(M,N,phi)
%   M     % the number of cells in the horizontal dimension
%   N     % the number of cells in the vertical dimension
%   phi   % a 2-by-1 vector parameterizing exponential kernel function
%                    k(i,j) = phi(2)^2 * exp(-||iPos-jPos||/phi(1))
%
% Returns a structure GP with fields
%   Param % a 4-by-1 vector of the input arguments [M;N;phi]
%   Coord % an (M*N)-by-2 matrix of spatial coordinates (x,y) corresponding 
%         % to the cell centers; perfectly square grids occupy the 2-d 
%         % region [0,1]x[0,1], while non-square grids force the dimension 
%         % with the greater number of cells to occupy [0,1] and the other
%         % dimension to occupy [0,f] for some f in (0,1) in proportion to 
%         % the ratio of M and N.
%   Sigma % an (M*N)-by-(M*N) covariance matrix with pairwise covariances 
%         % defined by the exponential kernel function of phi
%   Mu    % a length-(M*N) vector with component means, initialized as zero
%   Value % a length-(M*N) vector of known cell values, initialized as NaN

% Set default values
if nargin < 5, mumix = []; end
if nargin < 4, DEBUG = false; end
if nargin < 3, phi = [1; 1]; end
if nargin < 2, N = M; end

assert(M==floor(M)&&M>0&&N==floor(N)&&N>0,'Number of cells must be positive integer');
assert(phi(1)>=0,'Cross-cell correlation must be non-negative');
assert(phi(2)>0,'Per-cell standard deviation must be positive');

% Assign spatial coordinates (of cell centers) to the cell grid
if M > N % Non-square area-of-interest
  xMax = 1; yMax = N/M;
else
  yMax = 1; xMax = M/N;
end
xC = linspace(0,xMax,M+1); delta = diff(xC(1:2)); xC = xC(2:end) - delta/2; 
yC = linspace(0,yMax,N+1); yC = yC(2:end) - delta/2;
[xC,yC] = meshgrid(xC,yC); xC = xC'; yC = yC';

% Compute covariance matrix using exponential kernel function
n = M*N; Sigma = eye(n)*phi(2)^2;
if phi(1) > 0
  disp(['Constructing covariance matrix of length-' num2str(n) ' Gaussian process...']);
  for i = 1:n
    iPos = [xC(i); yC(i)]; % linearly index into meshgrid
    for j = i+1:n
      jPos = [xC(j); yC(j)]; % linearly index into meshgrid
      Sigma(i,j) = phi(2)^2*exp(-norm(iPos-jPos)/phi(1)); % exponential kernel 
      Sigma(j,i) = Sigma(i,j); % Covariance matrix is symmetric
    end
  end
end
Mu = zeros(n,1);
% Input MUMIX is also optional (with default value of []) and denotes the 
% Gaussian mixture parameters to construct a non-zero mean-field (i.e., 
% mean vector Mu). Specifically, each r'th row of MUMIX corresponds to a 
% component bivariate Gaussian PDF over the 2-D region
%   -- centered at mean vector MUMIX(r,2:3)';
%   -- rotated by covariance matrix [MUMIX(r,4), MUMIX(r,6); MUMIX(r,6), MUMIX(r,5)]; and 
%   -- weighted by MUMIX(r,1)
% The mean field becomes the sum of these weighted bell curves.
for c = 1:size(mumix,1)
  wc = mumix(c,1); mc = mumix(c,2:3)'; s12 = prod(mumix(c,4:6)); 
  Kc = inv(diag(mumix(c,4:5).^2) + s12*fliplr(eye(2)));
  for i = 1:n
    iPos = [xC(i); yC(i)];
    Mu(i) = Mu(i) + wc*exp(-0.5*(iPos-mc)'*Kc*(iPos-mc));
  end
end

GP.Param = [M;N;phi];
GP.MuMix = mumix;
GP.Coord = [xC(:) yC(:)];
GP.Sigma = Sigma;
GP.Mu = Mu;
GP.Value = nan(n,1);
% Some visualizations for debugging
if DEBUG
  subplot(2,2,1); showField(GP,GP.Mu,'temp');
  hold on; plot(GP.Coord(:,1),GP.Coord(:,2),'k.');
  axis equal; xlabel('x'); ylabel('y'); 
  title('Mean Field (and cell centers)');
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %Visualization uses the following variables above, but via GP are
  %  M = GP.Param(1); N = GP.Param(2); phi = GP.Param(3:4); 
  %  xC = reshape(GP.Coord(:,1),M,N); yC = reshape(GP.Coord(:,2),M,N);
  %  delta = diff(xC(1:2)); xMax = max(xC(:))+delta/2; yMax = max(yC(:))+delta/2; n = M*N;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  xlim([0 xMax]+0.01*xMax*[-1 1]); ylim([0 yMax]+0.01*yMax*[-1 1]);
  text(GP.Coord(1,1),GP.Coord(1,2),'1','Color',[1 0 0],...
       'HorizontalAlignment','center','VerticalAlignment','top');
  text(GP.Coord(M,1),GP.Coord(M,2),num2str(M),'Color',[1 0 0],...
       'HorizontalAlignment','center','VerticalAlignment','top');
  text(GP.Coord(n-M+1,1),GP.Coord(n-M+1,2),num2str(n-M+1),'Color',[1 0 0],...
       'HorizontalAlignment','center','VerticalAlignment','bottom');
  text(GP.Coord(n,1),GP.Coord(n,2),num2str(n),'Color',[1 0 0],...
       'HorizontalAlignment','center','VerticalAlignment','bottom');
  hold off;
  subplot(2,2,3); d = linspace(0,min([1,5*phi(2)]),1001);
  plot(d,phi(2)^2*exp(-d/phi(1)),'k-');
  xlabel('relative distance'); ylabel('pairwise covariance'); 
  title(['Exponential Kernel (L=' num2str(phi(1)) ', \sigma=' num2str(phi(2)) ')']);
  hold on; d0 = delta; 
  while d0 < max(d), plot(d0*[1 1],ylim,'k--'); d0 = d0 + delta; end
  hold off;
  if ~exist ('OCTAVE_VERSION', 'builtin') % octave doesn't have this func
    iptsetpref('ImshowAxesVisible','on');
  end
  subplot(2,2,2); imshow(GP.Sigma,[]); 
  title('Covariance Matrix'); colorbar;
  invSig = inv(GP.Sigma); invSig = invSig + diag(nan(M*N,1));
  subplot(2,2,4); imshow(invSig,[]); 
  title('Concentration Matrix (omitting diagonal)'); colorbar;
end
