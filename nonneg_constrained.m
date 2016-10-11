function nonneg_constrained()
% min f(x)
% s.t. x >= 0

% ------- parameters setting --------
% Rosenbrock:    f_size_max=2;   delta_bar = 1;   phi_bar = 7/180*pi;  
f_size_min = 0.05;         % outer: 100,  inner: 1000
f_size_max = 2;                
dmu_min = -0.9; 
dmu_max = -0.01; 
delta_bar = 1;                 % simpler problem, large delta_bar
phi_bar = 5/180*pi;            

x = [4;4];
lam = [1;1];                   % two inequality constraints
mu = 1.0;                      % mu: 1 -> 0
x0 = [3;3]; 
% ---------- looking for b0, c0 ----------
[f,g,h] = objfun(x0);
[cineq, Gcineq, Hcineq] = confun(x0); 
b0 = cineq + ones(size(cineq));        % cineq(x0) - b0 < 0;  c0 > 0; 
c0 = ones(size(cineq)); 
% -------- main part for homotopy continuation --------
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
x
outer_iter
end


function [Homo, dHdx, dHdmu, K_1] = obj_homo(x, lam, mu, x0, b0, c0)

[f, g, h] = objfun(x);
Df = g; 
[cineq, Gcineq, Hcineq] = confun(x); 


grad_lag = g + Gcineq*lam; 

K_1 = (1-mu).*grad_lag + mu.*(x-x0); 

ineq = mu.*b0 - cineq; 
K_2 = -abs( ineq - lam ).^3 + ineq.*3 + lam.^3 - mu.*c0; 

Homo = [K_1;
        K_2]; 

% ------------- calculating gradient --------------  
square_ineq_lam = (ineq - lam).^2; 
dCubicdX = 3.* ( -Gcineq  ) * diag(square_ineq_lam); 
dCubicdmu = 3.* b0 .* square_ineq_lam; 
dCubicdlam = 3.*(-eye(length(lam))) * diag(square_ineq_lam);

ind = ineq < lam;
if any(ind)
   dCubicdX(ind) = -dCubicdX(ind); 
   dCubicdmu(ind) = -dCubicdmu(ind); 
   dCubicdlam(ind) = -dCubicdlam(ind); 
end

hess_lag = h; 
for j = 1:length(cineq)
    hess_lag = hess_lag + lam(j).*Hcineq{j};
end


dK1dx = (1-mu).*hess_lag + mu.*eye(length(x)); 
dK2dx = -dCubicdX + 3.*(-Gcineq)* diag((ineq).^2); 
dK1dlam = (1-mu).*Gcineq; 
dK2dlam = -dCubicdlam + diag(3.*lam.^2); 

dHdx = [dK1dx, dK1dlam;
        dK2dx, dK2dlam]; 
    
dK1dmu = -grad_lag + (x-x0); 
dK2dmu = -dCubicdmu + 3.* b0 .* (ineq).^2 - c0; 
    
dHdmu = [dK1dmu; dK2dmu];  

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

% if Df > x
%     dCubicdX = 3.*(Df - x).^2' * (h - eye(length(x))); 
% elseif Df < x
%     dCubicdX = -3.*(Df - x).^2' * (h - eye(length(x))); 
% else
%     sprintf('Df == x! Cubic gradient not exist at middle absolute point')
%     pause
% end

%{

function [Homo, dHdx, dHdmu] = obj_homo(x, mu, a)
% 2D problem
% min ( (x+1)^2 + (y+1)^2 ) /2                convex in first try
% s.t. x >= 0   y >= 0   
% equivalent to:
% x >= 0;    Df >= 0;    x*Df = 0
% K = -|Df - x|^3 + Df^3 + x^3; 
% H = mu*K  + (1-mu)*(x-a)

[f, g, h] = objfun(x);
Df = g; 
K = -abs(Df - x).^3 + Df.^3 + x.^3; 

Homo = (1-mu)*K  + mu*(x-a); 

dCubicdX = 3.* (h - eye(length(x))) * (Df - x).^2; 
ind = Df < x; 
dCubicdX(ind) = -dCubicdX(ind); 

% if Df > x
%     dCubicdX = 3.*(Df - x).^2' * (h - eye(length(x))); 
% elseif Df < x
%     dCubicdX = -3.*(Df - x).^2' * (h - eye(length(x))); 
% else
%     sprintf('Df == x! Cubic gradient not exist at middle absolute point')
%     pause
% end

dKdx = -dCubicdX + 3*h*Df.^2 + 3*x.^2;
dHdx = (1-mu)*dKdx + mu; 
dHdmu = -K + (x-a); 

end

function [f,g,h] = objfun(x)
f = (x(1)+1)^2/2  + (x(2)+1)^2/2; 
g = [x(1)+1;
     x(2)+1];
h = [1,0; 0,1];  
end

%}

%{
function [Homo, dHdx, dHdmu] = obj_homo(x, mu, a)
% 1D problem
% min (x+1)^2/2       convex in first try
% s.t. x >= 0
% equivalent to:
% x >= 0;    Df >= 0;    x*Df = 0
% K = -|Df - x|^3 + Df^3 + x^3; 
% H = mu*K  + (1-mu)*(x-a)

[f, g, h] = objfun(x);
Df = g; 
K = -abs(Df - x)^3 + Df^3 + x^3; 

Homo = (1-mu)*K  + mu*(x-a); 

dCubicdX = 0; 
if Df > x
    dCubicdX = 3*(Df - x)^2 * (h - 1); 
elseif Df < x
    dCubicdX = -3*(Df - x)^2 * (h - 1); 
else
    sprintf('Df == x! Cubic gradient not exist at middle absolute point')
    pause
end

dKdx = -dCubicdX + 3*Df^2*h + 3*x^2;
dHdx = (1-mu)*dKdx + mu; 
dHdmu = -K + (x-a); 

end

function [f,g,h] = objfun(x)
f = (x-1)^2/2; 
g = x-1; 
h = 1; 
end
%}
