function path_following()










end

function [f,g,h] = objfun(x)
% solution
% x = [-9.5473    1.0474]
% f = 0.0236
f = exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);
g = zeros(2,1);
h = zeros(2,2);
g(1,1) = exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1) + ...
    exp(x(1))*(8*x(1) + 4*x(2)); 
g(2,1) = exp(x(1))*(4*x(2) + 4*x(1) +2);

h(1,1) = g(1,1) + exp(x(1))*(8*x(1) + 4*x(2)) + exp(x(1))*8;
h(1,2) = g(2,1) + exp(x(1))*(4);
h(2,1) = g(2,1) + exp(x(1))*(4);
h(2,2) = exp(x(1))*(4);

end

function [cineq, ceq, Gcineq, Gceq, vargout] = coninequ(x)
% Nonlinear inequality constraints
cineq = [1.5 + x(1)*x(2) - x(1) - x(2);     
     -x(1)*x(2) - 10];
 
dC1dx = [x(2)-1;
         x(1)-1]; 
dC2dx = [-x(2)
         -x(1)];
Gcineq = [dC1dx, dC2dx]; 

Hcineq = cell(1, length(cineq)); 
Hcineq{1} = [0,1;1,0];
Hcineq{2} = [0,-1;-1,0];
vargout = Hcineq; 

% Nonlinear equality constraints
ceq = [];
Gceq = []; 
end