function fs = hsmooth(f,n)
X = bsxfun(@minus, (1+n:length(f)+n)', (n-1:-1:0));
y = [ repmat(f(1),n,1); f ]; 
fs = mean(y(X),2);
end

