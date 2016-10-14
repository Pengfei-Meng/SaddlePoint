function path_following_unconstrained()
% working fine for unconstrained rosenbrock problem

n = 2;
x = [0;0];

mu = 1.0;
x0 = [-1;-1]; 
 
% ------- parameters setting --------
% Rosenbrock:    f_size_max=2;   delta_bar = 1;   phi_bar = 7/180*pi;  
f_size_min = 0.05;         % outer: 100,  inner: 1000
f_size_max = 2;                
dmu_min = -0.9; 
dmu_max = -0.01; 
delta_bar = 1;
phi_bar = 7/180*pi;               % can be manipulated here 
% -----------------------------------


[f, g, h] = objfun(x);
residual = (1-mu).*g + mu.*(x-x0);

norm0 = norm(g);
outer_tol = norm0*1e-6; 
step_size = 0.05;                 % norm(x0)  %2.0; 

iter = 0; 
inner_iter = 0; 
while norm0 > outer_tol  
    iter = iter + 1; 
    % predictor direction
    [f, g, h] = objfun(x);
    K = mu.*eye(n) + (1-mu).*h;
    b = (-g + (x - x0));
    dxdmu = K\b;
    tau = [dxdmu; -1];    
    t = tau./norm(tau);             % normalized Newton step
  
    if iter > 1
        % step length calculation
        delta = norm(x - x_p0); 
        phi = acos(t'*tsave); 
        
        f_size = max(sqrt(delta/delta_bar), phi/phi_bar); 
        f_size = max(f_size_min, f_size); 
        f_size = min(f_size_max, f_size); 
        step_size = step_size/f_size; 

        if f_size >= f_size_max
            x = xsave; 
            t = tsave; 
        end        
    end
    
    tsave = t; 
    xsave = x;  
    
    x = x + step_size.*t(1:end-1);
    
    dmu = step_size.*t(end); 
    dmu = max(dmu_min, dmu);
    dmu = min(dmu_max, dmu);    
    mu = mu + dmu; 
    
    mu = max(0.0, mu); 
        
    [f, g, h] = objfun(x);
    residual = (1-mu).*g + mu.*(x-x0);
    normr = norm(residual);
    inner_tol = normr*0.01;
     
    x_p0 = x;     
     % corrector
     while normr > inner_tol 
        inner_iter = inner_iter + 1; 
        K_newton = (1-mu).*h + mu.*eye(n);
        b_newton = -((1-mu)*g + mu*(x-x0));
        dx = K_newton\b_newton;
        
        x = x + dx;
        [f, g, h] = objfun(x);        
        residual = (1-mu).*g + mu.*(x-x0);
        normr = norm(residual); 
     end  
     
     norm0 = norm(g);   
end
iter
inner_iter
x
[f, g, h] = objfun(x);        
residual = (1-mu).*g + mu.*(x-x0);
normr = norm(g)

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


 
% function [f,g,h] = objfun(x)
% % solution
% % x = [0.5    -1.0]
% f = exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);
% g = zeros(2,1);
% h = zeros(2,2);
% g(1,1) = exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1) + ...
%     exp(x(1))*(8*x(1) + 4*x(2)); 
% g(2,1) = exp(x(1))*(4*x(2) + 4*x(1) +2);
% 
% h(1,1) = g(1,1) + exp(x(1))*(8*x(1) + 4*x(2)) + exp(x(1))*8;
% h(1,2) = g(2,1) + exp(x(1))*(4);
% h(2,1) = g(2,1) + exp(x(1))*(4);
% h(2,2) = exp(x(1))*(4);
% 
% end
 

function [f, g, h] = objfun(x)
% Rosenbrock function
% solution x=[1,1], f = 0; 
f = 100*(x(1)^2 - x(2))^2 + (x(1)-1)^2;

g = zeros(2,1); 
g(1) = 100*(2*(x(1)^2-x(2))*2*x(1)) + 2*(x(1)-1);
g(2) = 100*(-2*(x(1)^2-x(2)));
h=[-400*(x(2)-3*x(1)^2)+2, -400*x(1); -400*x(1), 200];
end
