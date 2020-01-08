function fprime = df(f)
n = length(f);
x = 1:n;
fp = [ f(1); f(1); f; f(n); f(n) ];
fprime = (fp(x+4) + 2*fp(x+3) - 2*fp(x+1) - fp(x))/8;
end
