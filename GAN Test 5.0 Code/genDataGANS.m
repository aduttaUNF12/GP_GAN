% genDataGANS.m
clear all;

% Gaussian process parameters
M = 32; N = 32; beta = 1.0; ell = 1.0; % see generateGP.m
%mumix = []; % Unstructured (zero) mean
mumix = [5,0.5,0.25,0.25,0.5,0; -5,0.5,0.75,0.25,0.5,0]; % Cool-top/Hot-bottom mean
%mumix = rand(2+randi(5),6); mumix(:,1) = 5*beta*(2*mumix(:,1)-1);
%mumix(:,4:5) = mumix(:,4:5)*ell/2; 
%mumix(:,6) = (2*mumix(:,6)-1).*sqrt(prod(mumix(:,4:5),2));
sigma_n = 0.5; % standard deviation of zero-mean sensor noise

% Path planning parameters
B = 30;        % budget (positive integer)
gamma = 0.0;   % in unit interval (0 = full exploitation & 1 = full exploration)
s = [2,3];     % start cell as 2-D index == 1-D index of (s(2)-1)*M + s(1)

% Run parameters
D = 3000;       % Number of data sets to generate (1 runs interactively with
                %                                  fixed starting location)
if D > 1
  GP = generateGP(M,N,[beta; ell],false,mumix);
else
  figure(1); GP = generateGP(M,N,[beta; ell],true,mumix);
end

% Initialize environment & path
TD = cell(D,1);
for d = 1:D
  if D > 1
    Z = sampleGP(GP); s = [randi(M),randi(N)];
  else
    figure(2); subplot(2,2,[1 3]); Z = sampleGP(GP,true);
  end
  Y = Z + sigma_n*randn(size(Z)); % Generate measurement field

  J = nan(B+1,3); J(1,1) = norm(Z-GP.Mu); % Initial RMSE
  J(1,2) = 0.5*(M*N + M*N*log(2*pi) + log(det(GP.Sigma))); % Initial exact entropy
  J(1,3) = 0.5*(M*N + M*N*log(2*pi) + log(det(diag(diag(GP.Sigma))))); % Initial entropy bound
  P = nan(B+1,1); P(1) = (s(2)-1)*M+s(1); % Initial path
  if D==1
    hold on; plot(GP.Coord(P(1),1),GP.Coord(P(1),2),'k-^'); hold off;
    subplot(2,2,2); plot(0:B,J(:,1),'k-o'); xlim([0 B]+[-1,1]*0.02*B);
    xlabel('stage'); ylabel('RMSE'); 
    subplot(2,2,4); plot(0:B,J(:,2:3),'-o'); xlim([0 B]+[-1,1]*0.02*B);
    xlabel('stage'); ylabel('Entropy'); legend('exact','bound') 
    pause;
  else
    disp(['  Generating sample path ' num2str(d) ' of ' num2str(D) '...']); 
  end
  
  GPd = GP;
  for t = 1:B
    llY = log(1/(sqrt(2*pi)*sigma_n)*exp(-0.5*(Y(P(t))-GPd.Mu(P(t)))^2/sigma_n^2));
    llM = log(1/(sqrt(2*pi)*sigma_n));
%    disp(['Log-likelihood of Y = ' num2str(llY) ' (relative to 0 of ' num2str(llM) ')'])
    [GPd.Mu,GPd.Sigma] = posteriorGP(GPd,[P(t),Y(P(t))],sigma_n^2); 
    P(t+1) = greedyAction(GPd,P(t),gamma);
    J(t+1,1) = norm(Z-GPd.Mu); 
    J(t+1,2:3) = 0.5*(M*N + M*N*log(2*pi) + log([det(GPd.Sigma) det(diag(diag(GPd.Sigma)))]));

    if D==1
      subplot(2,2,[1,3]); showField(GPd,Z,['t=' num2str(t) ': L(y) = ' num2str(llY) ' (while L(\mu) = ' num2str(llM) ')'],GPd.Mu);
      hold on;
      plot(GPd.Coord(P(1:t+1),1),GPd.Coord(P(1:t+1),2),'k-^');
      for k = 1:t
        yBar = exp(Y(P(k))) ./ (1+exp(Y(P(k))));
        yCol = [1 0 1] - [1 0 0]*2*max(0.5-yBar,0) - [0 0 1]*2*max(yBar-0.5,0);
        plot(GPd.Coord(P(k),1),GPd.Coord(P(k),2),'ko','MarkerSize',12,'MarkerFaceColor',yCol);
      end
      plot(GPd.Coord(P(t+1),1),GPd.Coord(P(t+1),2),'k-^');
      hold off;
      subplot(2,2,2); plot(0:B,J(:,1),'k-o'); xlim([0 B]+[-1,1]*0.02*B);
      xlabel('stage'); ylabel('RMSE'); 
      subplot(2,2,4); plot(0:B,J(:,2:3),'-o'); xlim([0 B]+[-1,1]*0.02*B);
      xlabel('stage'); ylabel('Entropy'); legend('exact','bound') 
      pause
    end
  end
  TD{d}.Z = Z; % length-MN state vector (linearly indexed into grid) 
  TD{d}.Y = Y; % length-MN measurement vector (linearly indexed into grid)
  TD{d}.P = P; % length-K path (sequence of linear indices)
  TD{d}.J = J; % length-K metrics [RMSE EntropyBnd Entropy]
end
gans.GP = GP;
gans.PP = [B sigma_n gamma s];
gans.TD = TD;
% The measurement field of the dth experiment is in TD{d}.Y, while the
% state field of the dth experiment is in Td{d}.Z
% -- They are both length-MN column vectors
% -- GP.Coord gives the corresponding (horz/vert) coordinates into grid
% Please follow the following convention to convert into 2D grid e.g., 
% -- Yin2D = rot90(reshape(TD{d}.Y,[],32))
