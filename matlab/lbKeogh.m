function distance = lbKeogh(seriesA, Ub, Lb)
% distance = 0;
% n = length(seriesA);
% 
% for i = 1:n
%     if seriesA(i) > Ub(i)
%         distance = distance + (seriesA(i)-Ub(i))^2;
%     elseif seriesA(i) < Lb(i)
%         distance = distance + (seriesA(i)-Lb(i))^2;
%     end
% end

% one liner code from https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm
% LB_Keogh = sqrt(sum([[seriesA > Ub].* [seriesA - Ub]; [seriesA < Lb].* [Lb - seriesA]].^2));
distance = sum([[seriesA > Ub].* [seriesA - Ub]; [seriesA < Lb].* [Lb - seriesA]].^2);

% or
% LB_Keogh = sqrt(sum([[Q > movmax(C,ww)].* [Q-movmax(C,ww)]; [Q < movmin(C,ww)].* [movmin(C,ww)-Q]].^2))

end