function test_K_slack()
% test the cubic K function on linear constrained problems
% min f(x)
% s.t. -(Ax-b) <= 0     lam > 0

f_size_min = 0.05;             
f_size_max = 5;                
dmu_min = -0.9; 
dmu_max = -0.01; 
delta_bar = 1;                 % simpler problem, large delta_bar
phi_bar = 5/180*pi;            

% Brown Zingg: mu: 1 -> 0 (used here);   Watson : mu: 0 -> 1
mu = 1.0;   
% x0 = [-1; 1];           
% s0 = [2; 2]; 
% lam0 = [0;0];
% x = [-3; -3];  
% s = [1; 1]; 
% lam = [1; 1];

% x0 = [-3; 3];  
% s0 = [5; 2]; 
% lam0 = [0;0];
% x = [-3; -3];  
% s = [2; 2]; 
% lam = [1; 1];

% Rosen-Suzuki
 
x0 = [1;1;1;1];
s0 = ones(3,1);
% lam0 = [0.1;0.1;0.1];
lam0 = zeros(3,1);
% x = [1;1;1;1];     
% s = [1;1;1]; 
% lam = ones(3,1);
x = x0;
s = s0;
lam = lam0;


nx = length(x);
ns = length(s); 

K0 = obj_K(x,s,lam, mu); 
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
figure('Name', 'Lam Path')
hold on 
scatter(x(1), x(2),[],'filled')
% scatter(lam(1), lam(2),[],'filled')


while mu > 0.0
     outer_iter = outer_iter + 1; 
     s; 
     % predictor direction
     [Homo, dHdx, dHdmu] = obj_homo(x, s, lam, mu, x0, s0, lam0); 
     
%      if norm(Homo) < 1e-5
%          break
%      end
     
     % dxdmu = gmres(dHdx, dHdmu, [], 0.5);
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
    
    x = x + step_size.*t(1:nx);
    s = s + step_size.*t(nx+1:nx+ns);
    lam = lam + step_size.*t(nx+ns+1:end-1);
    
    dmu = step_size.*t(end); 
    dmu = max(dmu_min, dmu);
    dmu = min(dmu_max, dmu);    
    mu = mu + dmu; 
    
    mu = max(0.0, mu); 
    
    [Homo, dHdx, dHdmu] = obj_homo(x, s, lam, mu, x0, s0, lam0); 
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
        x = x + dx(1:nx);
        s = s + dx(nx+1:nx+ns); 
        lam = lam + dx(nx+ns+1:end); 
                
        [Homo, dHdx, dHdmu] = obj_homo(x, s, lam, mu, x0, s0, lam0); 
        normH = norm(Homo);
    end 
    
    K = obj_K(x,s,lam, mu); 
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

function K = obj_K(x, s, lam, mu)
[f, df, hf] = objfun(x);
[g, dg, hg] = confun(x); 

lag_grad = df + dg*lam; 

e = ones(size(lam)); 
K = [lag_grad;
     s.*lam - mu.*e; 
     g + s]; 
end

function [Homo, dHdxsl, dHdmu] = obj_homo(x, s, lam, mu, x0, s0, lam0)
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
dK1ds = zeros(length(K1), length(s));
dK1dlam = (1-mu).*dg; 
dK1dmu = -lag_grad + (x-x0);

e = ones(size(lam)); 
% K2 = (1-mu).*(s.*lam - mu.*e) + mu.*(s-s0); 
% dK2dx = zeros(length(lam), length(x));
% dK2ds = (1-mu).*diag(lam) + mu.*eye(length(s));   
% dK2dlam = (1-mu).*diag(s) ;                       
% dK2dmu = -s.*lam + (2*mu-1).*e + (s-s0); 

K2 = (1-mu).*(s.*lam) + mu.*(s-s0); 
dK2dx = zeros(length(lam), length(x));
dK2ds = (1-mu).*diag(lam) + mu.*eye(length(s)); 
dK2dlam = (1-mu).*diag(s);    
dK2dmu = -s.*lam + (s-s0); 

% K2 = (1-mu).*(s.*lam) - mu.*e; 
% dK2dx = zeros(length(lam), length(x));
% dK2ds = (1-mu).*diag(lam./s); 
% dK2dlam = (1-mu).*eye(length(K2)); 
% dK2dmu = -s.*lam - e; 

K3 = (1-mu).*(g+s) - mu.*(lam - lam0); 
dK3dx = (1-mu).*dg'; 
dK3ds = (1-mu).*eye(length(K3)); 
dK3dlam = -mu.*eye(length(lam)); 
dK3dmu = -(g+s) - (lam-lam0); 

Homo = [K1;
        K2;
        K3];    
dHdxsl = [dK1dx, dK1ds, dK1dlam;
          dK2dx, dK2ds, dK2dlam; 
          dK3dx, dK3ds, dK3dlam];
dHdmu = [dK1dmu;
         dK2dmu
         dK3dmu]; 
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
% x0 = [0.8; 0.9];       % x0, is critical
% lam0 = [0.1;0.1];      % here g<=0
% x = [4; 3];   
% lam = [1; 1];

A = [1,0;
     0,1];
b = [0;0];

g = -(A*x - b); 
dg = -A; 
hg = cell(1, length(g)); 
hg{1} = zeros(length(x));
hg{2} = zeros(length(x));

end
%}



function [f,df,hf] = objfun(x)
%  Rosen-Suzuki Problem
%  min  x1^2 + x2^2 + 2*x3^2 + x4^2        - 5*x1 -5*x2 -21*x3 + 7*x4
%  s.t. 8  - x1^2 -   x2^2 - x3^2 -   x4^2 -   x1 + x2 - x3 + x4 >= 0 
%       10 - x1^2 - 2*x2^2 - x3^2 - 2*x4^2 +   x1           + x4 >= 0          
%       5- 2*x1^2 -   x2^2 - x3^2          - 2*x1 + x2      + x4 >= 0            
%  Initial Point x = [1,1,1,1];   
%  Solution at   x = [0,1,2,-1]; 
%                f = -44   
%  Common wrong solution x = [2.5000, 2.5000, 5.2500, -3.5000]
%                        f = -79.8750
f = x(1)^2 + x(2)^2 + 2*x(3)^2 + x(4)^2 -5*x(1) -5*x(2)-21*x(3) + 7*x(4); 

% its derivative wrt. x
df = zeros(4,1); 
df(1)= 2*x(1)-5;
df(2)= 2*x(2)-5;
df(3)= 4*x(3)-21;
df(4)= 2*x(4)+7; 

hf = diag([2,2,4,2]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [g, dg, hg]= confun(x)
% Constraint function
g = zeros(3,1);
g(1) = 8 - x(1)^2 -  x(2)^2-x(3)^2 - x(4)^2 - x(1) + x(2) - x(3) + x(4); 
g(2) = 10- x(1)^2 -2*x(2)^2-x(3)^2 - 2*x(4)^2 +   x(1)  + x(4)         ;
g(3) = 5-2*x(1)^2 -  x(2)^2-x(3)^2          - 2*x(1) + x(2)      + x(4);  

% Gradients of the constraint functions wrt. x
dg=[-2*x(1)-1, -2*x(2)+1, -2*x(3)-1, -2*x(4)+1; 
       -2*x(1)+1, -4*x(2),   -2*x(3),   -4*x(4)+1;
       -4*x(1)-2, -2*x(2)+1, -2*x(3),   1];
dg = dg'; 
ceq = [];  Gceq = []; 

Hcineq = cell(1, length(g)); 
Hcineq{1} = diag([-2, -2, -2, -2]);
Hcineq{2} = diag([-2, -4, -2, -4]);
Hcineq{3} = diag([-4, -2, -2,  0]);

% % if fmincon, or the interior points on paper:   c<0
g = -g;
dg = -dg; 
Hcineq{1} = -1*Hcineq{1}; 
Hcineq{2} = -1*Hcineq{2};
Hcineq{3} = -1*Hcineq{3};
hg = Hcineq; 
end


%{
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
%}