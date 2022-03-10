function vNext = greedyAction(GP,vThis,gamma)
% function vNext = greedyAction(GP,vThis,gamma)

M = GP.Param(1); N = GP.Param(2);
Hhat = 0.5*log(2*pi*exp(1)*diag(GP.Sigma)); % Per-cell entropies
uRew = -inf(1,4); % Rewards of going East, North, West, South
if mod(vThis,M), uRew(1) = Hhat(vThis+1); end % Can move east
if vThis+M <= M*N, uRew(2) = Hhat(vThis+M); end % Can move north
if mod(vThis-1,M), uRew(3) = Hhat(vThis-1); end % Can move west
if vThis-M >= 1, uRew(4) = Hhat(vThis-M); end % Can move south

% Action under exploitation
u0 = find(uRew==max(uRew)); u0 = u0(randi(length(u0),1));

% Action under exploration
pVec = zeros(size(uRew)); pVec(~isinf(uRew)) = uRew(~isinf(uRew)); 
pVec(~isinf(uRew)) = pVec(~isinf(uRew))-min(pVec(~isinf(uRew))); 
if sum(pVec)
  pVec = pVec / sum(pVec);
else
  pVec(~isinf(uRew)) = 1; pVec = pVec / sum(pVec);
end
u1 = find(rand(1,1)<=cumsum(pVec),1);

if rand > gamma, uStar = u0; else, uStar = u1; end
switch uStar
  case 1, vNext = vThis+1;
  case 2, vNext = vThis+M;
  case 3, vNext = vThis-1;
  case 4, vNext = vThis-M;
end
