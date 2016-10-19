function test_K_cubic2()
% test the cubic K function on linear constrained problems
% min f(x)
% s.t. -(Ax-b) <= 0     lam > 0
% it is working right now! 

f_size_min = 0.05;             
f_size_max = 5;                
dmu_min = -0.9; 
dmu_max = -0.01; 
delta_bar = 1;                 % simpler problem, large delta_bar
phi_bar = 6/180*pi;            

% Brown Zingg: mu: 1 -> 0 (used here);   Watson : mu: 0 -> 1
mu = 1.0;   
% x0 = [1; 1];       % x0, is critical
% lam0 = [0.1;0.1];
% b0 = [0.1; 0.1];
% c0 = [0.1; 0.1]; 

% polynomial
x0 = [0.8; 0.9];        
lam0 = [0.1;0.1];
b0 = [1;1];
c0 = [0.3; 0.3]; 

% x0 = [-7; 0.6];        
% lam0 = [2; 2];
% b0 = [1; 1];
% c0 = [0.3; 0.3]; 
x = [4; 3];   
lam = [1; 1];

repair = true; 

K0 = obj_K(x,lam); 
normK = norm(K0,Inf); 
K0_tol = normK*1e-6; 
% solving the Cubic homotopy using hometopy continuation 
% with mu 1 -> 0, the same method as in David Brown's paper
step_size = 0.05;               
outer_iter = 0; 
inner_iter = 0; 

x_hist = [];   lam_hist = []; 
x_hist = [x_hist, x];
lam_hist = [lam_hist, lam]; 
figure('Name', 'x Path')
hold on 
scatter(x(1), x(2),[],'filled')
% scatter(lam(1), lam(2),[],'filled')


while mu > 0.0
     outer_iter = outer_iter + 1; 
     % predictor direction
     if repair
        [Homo, dHdx, dHdmu]=obj_homo_repair(x, lam, mu, x0, lam0, b0, c0); 
     else
        [Homo, dHdx, dHdmu] = obj_homo(x, lam, mu, x0, lam0); 
     end
     
     if norm(Homo) < 1e-5
         break
     end
     
     % dxdmu = gmres(dHdx, dHdmu);
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

%         if f_size >= f_size_max
%             sprintf('backtracking')
%             x = xsave; 
%             t = tsave; 
%             lam = lamsave; 
%         end        
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
    
    if repair
        [Homo, dHdx, dHdmu]=obj_homo_repair(x, lam, mu, x0, lam0, b0, c0); 
    else
        [Homo, dHdx, dHdmu] = obj_homo(x, lam, mu, x0, lam0);
    end
    % lam = solve_lam(mu,x,b0,c0, lam);
    
    normH = norm(Homo);
    inner_tol = normH*0.1;     % 0.1 is best choice for cubic complementary
    
    x_p0 = x;     
     
    % corrector
    newton_iter = 0; 
    while (normH > inner_tol) && (newton_iter < 20)
        newton_iter = newton_iter + 1; 
        inner_iter = inner_iter + 1; 
        dx = -dHdx\Homo;
        % dx = -gmres(dHdx, Homo);
        x = x + dx(1:length(x));
        lam = lam + step_size.*t(length(x)+1:end-1);
        
        if repair
            [Homo, dHdx, dHdmu]=obj_homo_repair(x, lam, mu, x0, lam0, b0, c0); 
        else
            [Homo, dHdx, dHdmu] = obj_homo(x, lam, mu, x0, lam0);
        end
        normH = norm(Homo);
    end 
    
    K = obj_K(x,lam); 
    normK = norm(K,Inf); 

    if normK < K0_tol
        break
    end
    
    x_hist = [x_hist, x];
    lam_hist = [lam_hist, lam]; 
    hold on 
    scatter(x(1), x(2),[],'filled')
    
    % scatter(lam(1), lam(2),[],'filled')
    % updating a is not helping here... 
    
end

% c = linspace(1,10,size(x_hist,2));
% scatter(x_hist(1,:), x_hist(2,:), [],c,'filled')
x
mu
outer_iter
inner_iter
end

function K = obj_K(x,lam)
[f, df, hf] = objfun(x);
[g, dg, hg] = confun(x); 

lag_grad = df + dg*lam; 
K = -abs(g - lam).^3 + g.^3 + lam.^3;

K = [lag_grad;
           K]; 
end

function [Homo, dHdxl, dHdmu] = obj_homo(x, lam, mu, x0, lam0)
% for the simple problem
% K = - |df(x) - x| + df^3 + x^3 = 0; 

[f, df, hf] = objfun(x);
[g, dg, hg] = confun(x); 

lag_grad = df + dg*lam; 
lag_hess = hf; 
for j = 1:length(g)
    lag_hess = lag_hess + lam(j).*hg{j};
end

K1 = (1-mu).*lag_grad + mu.*(x-x0);
dK1dx = (1-mu).*lag_hess + mu.*eye(length(x));
dK1dlam = (1-mu).*dg; 
dK1dmu = -lag_grad + (x-x0);

K = -abs(g - lam).^3 + g.^3 + lam.^3; 
dKdx = -dg*diag(3.*(g-lam).^2 .*sign(g-lam)) + dg*diag(3.*g.^2); 
dKdlam =  diag(3.*(g-lam).^2 .*sign(g-lam)) + diag(3.*lam.^2);

K2 = (1-mu).*K + mu.*(lam - lam0);

dK2dx = (1-mu).*dKdx ; 
dK2dlam = (1-mu).*dKdlam  + mu.*eye(length(lam));
dK2dmu = -K + (lam - lam0);


Homo = [K1;
        K2];    
dHdxl = [dK1dx, dK1dlam;
         dK2dx, dK2dlam];
dHdmu = [dK1dmu;
         dK2dmu]; 
end


function [Homo, dHdxl, dHdmu]=obj_homo_repair(x, lam, mu, x0, lam0, b0, c0)
% for the simple problem
% K = - |df(x) - x| + df^3 + x^3 = 0; 

[f, df, hf] = objfun(x);
[g, dg, hg] = confun(x); 

lag_grad = df + dg*lam; 
lag_hess = hf; 
for j = 1:length(g)
    lag_hess = lag_hess + lam(j).*hg{j};
end

K1 = (1-mu).*lag_grad + mu.*(x-x0);
dK1dx = (1-mu).*lag_hess + mu.*eye(length(x));
dK1dlam = (1-mu).*dg; 
dK1dmu = -lag_grad + (x-x0);

K = -abs(g-mu.*b0 - lam).^3 + (g-mu.*b0).^3 + lam.^3 + mu.*c0; 
dKdx = -dg*diag(3.*(g-mu.*b0-lam).^2 .*sign(g-mu.*b0-lam)) + dg*diag(3.*(g-mu.*b0).^2); 
dKdlam =  diag(3.*(g-mu.*b0-lam).^2 .*sign(g-mu.*b0-lam)) + diag(3.*lam.^2);
dKdmu = b0.*(3.*(g-mu.*b0-lam).^2 .*sign(g-mu.*b0-lam)) - b0.*(3.*(g-mu.*b0).^2) + c0; 

% K2 = (1-mu).*K + mu.*(lam - lam0);
% dK2dx = (1-mu).*dKdx ; 
% dK2dlam = (1-mu).*dKdlam  + mu.*eye(length(lam));
K2 = K; 
dK2dx = dKdx; 
dK2dlam = dKdlam; 
dK2dmu = dKdmu; 

Homo = [K1;
        K2];    
dHdxl = [dK1dx, dK1dlam;
         dK2dx, dK2dlam];
dHdmu = [dK1dmu;
         dK2dmu]; 
end


%{
function [f,df,hf] = objfun(x)
% note: the constraint x >= 0 is assimilated into func: obj_homo

% initial x0, x is critical
% % first problem
% x0 = [2;2]; 
% x = [3; 3];     % find the solution to the 1e-4 precision

% f = 1/2*(x(1) + 1)^2  + 1/2*(x(2) + 1)^2;
% df = [x(1) + 1;
%      x(2) + 1];
% hf = [1,0;
%      0,1];  

 
% x0 = [0.6;4];       % x0, is critical
% x = [2; 2];         % strangely, this is not as robust as assumed
                      % can you find out the reason why?
                      % some x0, x will work; others doesn't? 
f = (x(1) + 1)*(x(1) - 2)  + (x(2) - 1)*(x(2) + 2);
df = [2*x(1)-1;
     2*x(2)+1];
hf = [2,0;
     0,2];

end

function [g, dg, hg] = confun(x)
% x0 = [0.8; 0.9];        
% lam0 = [0.1;0.1];
% x = [4; 3];   
% lam = [1; 1];

A = [2,0;
     0,3];
b = [1;1];

g = -(A*x - b); 
dg = -A; 
hg = cell(1, length(g)); 
hg{1} = zeros(length(x));
hg{2} = zeros(length(x));

end
%}


 
function [f,df,hf] = objfun(x)
% solution
% x = [-9.5473    1.0474]
% f = 0.0236
f = exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);
df = zeros(2,1);
hf = zeros(2,2);
df(1,1) = exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1) + ...
    exp(x(1))*(8*x(1) + 4*x(2)); 
df(2,1) = exp(x(1))*(4*x(2) + 4*x(1) +2);

hf(1,1) = df(1,1) + exp(x(1))*(8*x(1) + 4*x(2)) + exp(x(1))*8;
hf(1,2) = df(2,1) + exp(x(1))*(4);
hf(2,1) = df(2,1) + exp(x(1))*(4);
hf(2,2) = exp(x(1))*(4);

end

function [g, dg, hg] = confun(x)
% Nonlinear inequality constraints
g = [1.5 + x(1)*x(2) - x(1) - x(2);     
     -x(1)*x(2) - 10];
 
dC1dx = [x(2)-1;
         x(1)-1]; 
dC2dx = [-x(2)
         -x(1)];
dg = [dC1dx, dC2dx]; 

hg = cell(1, length(g)); 
hg{1} = [0,1;1,0];
hg{2} = [0,-1;-1,0];

% Nonlinear equality constraints
% ceq = x(1)^2 + x(2)^2 - 4;
% Gceq = [2*x(1);
%         2*x(2)];   
% Hceq = cell(1, length(ceq));    
% Hceq{1} = [2,0;
%            0,2]; 
end
 