function nonneg_constrained()
% min f(x)
% s.t. x >= 0

% ------- parameters setting --------
% Rosenbrock:    f_size_max=2;   delta_bar = 1;   phi_bar = 7/180*pi;  
f_size_min = 0.05;         % outer: 100,  inner: 1000
f_size_max = 5;                
dmu_min = -0.9; 
dmu_max = -0.01; 
delta_bar = 1;
phi_bar = 7/180*pi;               % can be manipulated here 

x = [4;4];
mu = 1.0;
x0 = [4;4]; 


% -------- main part for homotopy continuation --------
step_size = 0.05;               

outer_iter = 0; 
inner_iter = 0; 
while mu > 0.0
    outer_iter = outer_iter + 1; 
    
    % predictor direction
    [Homo, dHdx, dHdmu] = obj_homo(x, mu, x0); 
    
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
    
    [Homo, dHdx, dHdmu] = obj_homo(x, mu, x0);    
    normH = norm(Homo);  
    inner_tol = normH*0.01;
    
    x_p0 = x;     
     % corrector
     while normH > inner_tol 
        inner_iter = inner_iter + 1; 
        dx = -dHdx\Homo;
        
        x = x + dx;
        [Homo, dHdx, dHdmu] = obj_homo(x, mu, x0);    
        normH = norm(Homo); 
     end       
end
x
outer_iter
end


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
