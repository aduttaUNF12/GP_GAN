function showField(GP,Z,tStr,zHat)
% function showField(GP,Z,tStr,zHat)

if nargin < 4, zHat = nan(size(Z)); end

plot(GP.Coord(:,1),GP.Coord(:,2),'k.');
axis equal; xlabel('x'); ylabel('y'); title(tStr);
delta = diff(GP.Coord(1:2,1));
xMax = max(GP.Coord(:,1))+delta/2; yMax = max(GP.Coord(:,2))+delta/2; 
xlim([0 xMax]+0.01*xMax*[-1 1]); ylim([0 yMax]+0.01*yMax*[-1 1]);
ZBar = exp(Z) ./ (1+exp(Z));
zHatBar = exp(zHat) ./ (1+exp(zHat));
for i = 1:length(ZBar)
  theCol = [1 0 1] - [1 0 0]*2*max(0.5-ZBar(i),0) - [0 0 1]*2*max(ZBar(i)-0.5,0);
  patch([GP.Coord(i,1)-delta/2,GP.Coord(i,1)-delta/2,GP.Coord(i,1)+delta/2,GP.Coord(i,1)+delta/2],...
        [GP.Coord(i,2)-delta/2,GP.Coord(i,2)+delta/2,GP.Coord(i,2)+delta/2,GP.Coord(i,2)-delta/2],theCol);
  if ~isnan(GP.Value(i))
    hold on; plot(GP.Coord(i,1),GP.Coord(i,2),'k.'); hold off; 
  elseif ~isnan(zHatBar(i))
    theCol = [1 0 1] - [1 0 0]*2*max(0.5-zHatBar(i),0) - [0 0 1]*2*max(zHatBar(i)-0.5,0);
      patch([GP.Coord(i,1)-delta/4,GP.Coord(i,1)-delta/4,GP.Coord(i,1)+delta/4,GP.Coord(i,1)+delta/4],...
            [GP.Coord(i,2)-delta/4,GP.Coord(i,2)+delta/4,GP.Coord(i,2)+delta/4,GP.Coord(i,2)-delta/4],theCol);    
  end
end
