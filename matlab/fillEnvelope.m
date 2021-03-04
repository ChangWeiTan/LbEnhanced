function [U,L] = fillEnvelope(series, W)
U = zeros(size(series));
L = zeros(size(series));

for i = 1:length(series)
    start = max(i-W, 1);
    stop = min(i+W, length(series));
    U(i) = max(series(start:stop));
    L(i) = min(series(start:stop));
end
end