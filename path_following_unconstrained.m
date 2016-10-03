function path_following_unconstrained()

n = 2;
x = [1;1];

mu = 1.0;
% x0 = [0;0];
x0 = x.*0.8; 

step_size = norm(x);           % this is pure guessing
[f, g, h] = objfun(x);
residual = (1-mu).*g + mu.*(x-x0);
norm0 = norm(residual);
normr = norm0;
xold = x; 
told = zeros(n+1,1); 

delta_bar = 1.4;
phi_bar = 1.3; 

while mu >= 0    
    % corrector
    while normr > norm0*0.3                
        K_newton = (1-mu).*h + mu.*eye(n);
        b_newton = -((1-mu)*g + mu*(x-x0));
        dx = K_newton\b_newton;
        
        x = x + dx;
        [f, g, h] = objfun(x);        
        residual = (1-mu).*g + mu.*(x-x0);
        normr = norm(residual); 
    end  

    % predictor direction
    [f, g, h] = objfun(x);
    K = mu.*eye(n) + (1-mu).*h;
    b = -g + (x - x0);
    dxdmu = K\b;
    tau = [dxdmu; -1];
    t = tau./norm(tau);
    
    % step length calculation
    delta = norm(x - xold); 
    phi = acos(t'*told); 
    f = max(sqrt(delta/delta_bar), sqrt(phi/phi_bar)); 
    step_size = step_size/f; 
    
    % update predictor, mu, x
    x = x + step_size.*t(1:end-1);
    mu = mu + step_size.*t(end);   
    
    mu = 0.5*mu; 
end
x
end


% function [f, g, h] = objfun(x)
% % solution
% % x = [2.2500, -4.7500]
% % f = -16.3750
% 
% f = 3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2);
% 
% g = zeros(2,1);
% g(1) = 6*x(1) + 2*x(2) - 4;
% g(2) = 2*x(1) + 2*x(2) + 5;
% 
% h = [6,2;
%     2,2];
% end


function [f, g, h] = objfun(x)
% solution x=[1,1], f = 0; 
f = 100*(x(1)^2 - x(2))^2 + (x(1)-1)^2;

g = zeros(2,1); 
g(1) = 100*(2*(x(1)^2-x(2))*2*x(1)) + 2*(x(1)-1);
g(2) = 100*(-2*(x(1)^2-x(2)));
h=[-400*(x(2)-3*x(1)^2)+2, -400*x(1); -400*x(1), 200];
end
