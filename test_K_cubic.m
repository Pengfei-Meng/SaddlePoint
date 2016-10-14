% use Newton's method to solve
% K = - |df(x) - x| + df^3 + x^3 = 0; 
% 
% same solution as:
% min  f(x)   s.t. x >= 0 

function test_K_cubic()
    c = 0;
    x = fsolve(@(x) myfun(x,c),3)
end

function F = myfun(x,c)
    [f,df] = objfun(c);
    F = -abs(df-x).^3 + df^3 + x^3; 
end

function [f,df] = objfun(c)
% 1/2*(c+1)^2;
f = 1/2*(c+1)^2;
df = c+1; 
end