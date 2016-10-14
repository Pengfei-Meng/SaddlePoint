% use Homotopy method to solve
% K = - |df(x) - x| + df^3 + x^3 = 0; 
% 
% same solution as:
% min  f(x)   s.t. x >= 0 


function test_K_cubic()

f_size_min = 0.05;             
f_size_max = 5;                
dmu_min = -0.9; 
dmu_max = -0.01; 
delta_bar = 1;                 % simpler problem, large delta_bar
phi_bar = 5/180*pi;            

% Brown Zingg: mu: 1 -> 0 (used here);   Watson : mu: 0 -> 1
mu = 1.0;   
x0 = [9;9];
a = x0; 
x = [2; 2];
n = 2;      % number of design
m = 2;      % number of constraints


% solving the Cubic homotopy using hometopy continuation 
% with mu 1 -> 0, the same method as in David Brown's paper
step_size = 0.05;               
outer_iter = 0; 
inner_iter = 0; 

while mu > 0.0
     outer_iter = outer_iter + 1; 
     % predictor direction
     [Homo, dHdx, dHdmu] = obj_homo(x, mu, a); 
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
            sprintf('backtracking')
            x = xsave; 
            t = tsave; 
            % lam = lamsave; 
        end        
    end
    
    tsave = t; 
    xsave = x; 
    % lamsave = lam; 
    
    x = x + step_size.*t(1:length(x));
    % lam = lam + step_size.*t(length(x)+1:end-1);
    
    dmu = step_size.*t(end); 
    dmu = max(dmu_min, dmu);
    dmu = min(dmu_max, dmu);    
    mu = mu + dmu; 
    
    mu = max(0.0, mu); 
    
    [Homo, dHdx, dHdmu] = obj_homo(x, mu, a);
    % lam = solve_lam(mu,x,b0,c0, lam);
    
    normH = norm(Homo);
    inner_tol = normH*0.5;     % 0.5 is best choice for cubic complementary
    
    x_p0 = x;     
     
    % corrector
    while normH > inner_tol 
        inner_iter = inner_iter + 1; 
        dx = -dHdx\Homo;
        
        x = x + dx(1:length(x));
        
        [Homo, dHdx, dHdmu] = obj_homo(x, mu, a);
        normH = norm(Homo);
    end 
   
    if mu > 1e-2
        [Homo, dHdx, dHdmu, K] = obj_homo(x, mu, a);
        a = ((1-mu).*K + mu.*x)./mu  
    end
end
x
mu
outer_iter
inner_iter
end

function [Homo, dHdx, dHdmu, K] = obj_homo(x, mu, a)

[f,df,hf] = objfun(x);

K = -abs(df - x).^3 + df.^3 + x.^3; 

const1 = 3.*(df - x).^2.*sign(df-x); 
const2 = 3.*df.^2; 
dKdx = -(hf - eye(length(x)))*diag(const1) + hf*diag(const2); 

Homo = (1-mu).*K + mu.*(x-a); 
dHdx = (1-mu).*dKdx + mu.*eye(length(x));

dHdmu = -K + (x-a); 

end

function [f,g,h] = objfun(x)
% note: the constraint x >= 0 is assimilated into func: obj_homo
f = 1/2*(x(1) + 1)^2  + 1/2*(x(2) + 1)^2;
g = [x(1) + 1;
     x(2) + 1];
h = [1,0;
     0,1];  
end



%{
% fsolve can solve it for lambda, if c >= 0; here x : lambda, the unknown
% K = - |df(x) - x| + df^3 + x^3 = 0;  

function test_K_cubic()
    c = 1;
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
%}