function nonlinear_constrained()
% min exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);
% s.t. g = cineq = [1.5 + x(1)*x(2) - x(1) - x(2);     
%                   -x(1)*x(2) - 10];  
% g = cineq <= 0

f_size_min = 0.05;             
f_size_max = 2;                
dmu_min = -0.9; 
dmu_max = -0.01; 
delta_bar = 1;                 % simpler problem, large delta_bar
phi_bar = 5/180*pi;            

% Brown Zingg: mu: 1 -> 0 (used here);   Watson : mu: 0 -> 1
mu = 1.0;   
x = [4;4];           %   starting point
lam = [1;1];

n = 2;      % number of design
m = 2;      % number of constraints
[x0,b0,c0] = select_initials(n,m); 
a = x0; 

% solving the Cubic homotopy using hometopy continuation 
% with mu 1 -> 0, the same method as in David Brown's paper
step_size = 0.05;               
outer_iter = 0; 
inner_iter = 0; 

while mu > 0.0
     outer_iter = outer_iter + 1; 
     % predictor direction
     [Homo, dHdx, dHdmu, K1] = obj_homo(x, lam, mu, x0, b0, c0); 
         dxdmu = dHdx \ dHdmu;
    tau = [dxdmu; -1];    
    t = tau./norm(tau);             % normalized Newton step
  
    if outer_iter > 1
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
            lam = lamsave; 
        end        
    end
    
    tsave = t; 
    xsave = x; 
    lamsave = lam; 
    
    x = x + step_size.*t(1:length(x));
    lam = lam + step_size.*t(length(x)+1:end-1);
    
    dmu = step_size.*t(end); 
    dmu = max(dmu_min, dmu);
    dmu = min(dmu_max, dmu);    
    mu = mu + dmu; 
    
    mu = max(0.0, mu); 
    
    [Homo, dHdx, dHdmu, K1] = obj_homo(x, lam, mu, x0, b0, c0);
    
    normH = norm(K1);  
    inner_tol = normH*0.01;
    
    x_p0 = x;     
     
    % corrector
    while normH > inner_tol 
        inner_iter = inner_iter + 1; 
        dx = -dHdx\Homo;
        
        x = x + dx(1:length(x));
        lam = lam + dx(length(x)+1:end); 
        [Homo, dHdx, dHdmu, K1] = obj_homo(x, lam, mu, x0, b0, c0);
        normH = norm(K1); 
    end          
end


end

function [Homo, dHdx, dHdmu, K_1] = obj_homo(x, lam, mu, x0, b0, c0)

[f, Df, Hf] = objfun(x);
[g, Dg, Hg] = confun(x);       %  g <= 0  here

grad_lag = Df + Dg*lam; 
hess_lag = Hf; 
for j = 1:length(g)
    hess_lag = hess_lag + lam(j).*Hg{j};
end

% K_1 block

K_1 = (1-mu).*grad_lag + mu.*(x-x0); 

dK1dx = (1-mu).*hess_lag + mu.*eye(length(x)); 
dK1dlam = (1-mu).*Dg;           % multipliers
dK1dmu = -grad_lag + (x-x0);    % barrier parameter


% K_2 block

ineq = mu.*b0 - g; 
K_2 = -abs( ineq - lam ).^3 + ineq.*3 + lam.^3 - mu.*c0; 

% 1) dCubic w.r.t. x, lam, mu
square_ineq_lam = 3.*(ineq - lam).^2; 
dCubicdX = ( -Dg  ) * diag(square_ineq_lam); 
dCubicdmu = b0 .* square_ineq_lam; 
dCubicdlam = (-eye(length(lam))) * diag(square_ineq_lam);

ind = ineq < lam;
if any(ind)
   dCubicdX(ind) = -dCubicdX(ind); 
   dCubicdlam(ind) = -dCubicdlam(ind); 
   dCubicdmu(ind) = -dCubicdmu(ind); 
end

% 2) assemble dCubic into dK2
dK2dx = -dCubicdX + 3.*(-Dg)* diag((ineq).^2); 
dK2dlam = -dCubicdlam + diag(3.*lam.^2); 
dK2dmu = -dCubicdmu + 3.* b0 .* (ineq).^2 - c0; 

Homo = [K_1;
        K_2]; 

dHdx = [dK1dx, dK1dlam;
        dK2dx, dK2dlam]; 

dHdmu = [dK1dmu; dK2dmu];   

end  


function [x0,b0,c0] = select_initials(n,m)
% rules                              dimension
% x0 : arbitrary                     n 
% b0 : b0 > 0,   g(x0) - b0 <= 0     m
% c0 : c0 > 0                        m
x0 = [4; 0.5];     % so that cineq < 0 
[cineq, Gcineq, Hcineq] = confun(x0); 
b0 = ones(m,1); 
c0 = 0.5.*ones(m,1);
end

function [f,g,h] = objfun(x)
% A more complex case, only inequality constraints
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

function [cineq, Gcineq, Hcineq] = confun(x)
% Nonlinear inequality constraints   <= 0 here
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

end