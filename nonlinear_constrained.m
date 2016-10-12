function nonlinear_constrained()


f_size_min = 0.05;             
f_size_max = 2;                
dmu_min = -0.9; 
dmu_max = -0.01; 
delta_bar = 1;                 % simpler problem, large delta_bar
phi_bar = 5/180*pi;            

% Brown Zingg: mu: 1 -> 0 (used here);   Watson : mu: 0 -> 1
mu = 1.0;   
x = [1;1;1;1];           %   starting point

n = 4;      % number of design
m = 3;      % number of constraints
[x0,b0,c0] = select_initials(n,m); 
lam = 0.1.*ones(m,1);
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

Q = mu.*b0 - g; 
K_2 = -abs( Q - lam ).^3 + Q.*3 + lam.^3 - mu.*c0; 

%% 1) dCubic w.r.t. x, lam, mu
% square_ineq_lam = 3.*(ineq - lam).^2; 
% dCubicdX = ( -Dg  ) * diag(square_ineq_lam); 
% dCubicdmu = b0 .* square_ineq_lam; 
% dCubicdlam = (-eye(length(lam))) * diag(square_ineq_lam);
% 
% ind = ineq < lam;
% if any(ind)
%    dCubicdX(ind) = -dCubicdX(ind); 
%    dCubicdlam(ind) = -dCubicdlam(ind); 
%    dCubicdmu(ind) = -dCubicdmu(ind); 
% end
% 
% % 2) assemble dCubic into dK2
% dK2dx = -dCubicdX + 3.*(-Dg)* diag((ineq).^2); 
% dK2dlam = -dCubicdlam + diag(3.*lam.^2); 
% dK2dmu = -dCubicdmu + 3.* b0 .* (ineq).^2 - c0; 

%---------------- dK2dx, dK2dlam, dK2dmu --------------
% Q: mu.*b0 - g; 
% lam: multipliers

coef1_6lQ = 6.*lam.*Q; 
coef2_3l2 = 3.*lam.^2; 

coef3_3Q2 = 3.*Q.^2; 
coef4_6Q  = 6.*Q; 

dK2dx = zeros(length(x), length(K_2));
dK2dlam = zeros(length(K_2), length(K_2));
dK2dmu = zeros(length(K_2), 1);

ind_l = Q >= lam;
if any(ind_l)
    dK2dx(:,ind_l) = (-Dg(:,ind_l))*diag(coef1_6lQ(ind_l)) - ...
                     (-Dg(:,ind_l))*diag(coef2_3l2(ind_l)); 
    
    dK2dmu(ind_l) = diag(coef1_6lQ(ind_l))*b0(ind_l) - ...
                    diag(coef2_3l2(ind_l))*b0(ind_l) - c0(ind_l);
        
    dK2dlam(ind_l,ind_l) = diag(coef3_3Q2(ind_l)) - ...
                           diag(coef4_6Q(ind_l).*lam(ind_l)) + ...
                           diag(6.*lam(ind_l).^2); 
end

ind_s = Q < lam; 
if any(ind_s)
    dK2dx(:,ind_s) = (-Dg(:,ind_s))*diag(2.*coef3_3Q2(ind_s)) - ...
                     (-Dg(:,ind_s))*diag(coef1_6lQ(ind_s)) + ...
                     (-Dg(:,ind_s))*diag(coef2_3l2(ind_s)); 
                 
    dK2dmu(ind_s) = diag(2.*coef3_3Q2(ind_s))*b0(ind_s) - ...
                    diag(coef1_6lQ(ind_s))*b0(ind_s) + ...
                    diag(coef2_3l2(ind_s))*b0(ind_s) - c0(ind_s);
                                 
    dK2dlam(ind_s, ind_s) = -diag(coef3_3Q2(ind_s)) + ...
                             diag(coef4_6Q(ind_s).*lam(ind_s)); 
        
end


%------------------------------------------------------
Homo = [K_1;
        K_2]; 

dK2dx = dK2dx'; 
dHdx = [dK1dx, dK1dlam;
        dK2dx, dK2dlam]; 

cond(dHdx)
if cond(dHdx) > 1e12
   sprintf('condition number too high! %e', cond(dHdx))
   x
   lam
   pause 
end

dHdmu = [dK1dmu; dK2dmu];   

end  




%{
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
% min exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);
% s.t. g = cineq = [1.5 + x(1)*x(2) - x(1) - x(2);     
%                   -x(1)*x(2) - 10];  
% g = cineq <= 0
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
%}

function [x0,b0,c0] = select_initials(n,m)
% rules                              dimension
% x0 : arbitrary                     n 
% b0 : b0 > 0,   g(x0) - b0 <= 0     m
% c0 : c0 > 0                        m
x0 = [-1; -1; -1; -1];     % so that cineq < 0 
[cineq, Gcineq, Hcineq] = confun(x0); 
b0 = 0.5.*ones(m,1); 
c0 = 0.5.*ones(m,1);
end

function [f,g,h] = objfun(x)
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
g = zeros(4,1); 
g(1)= 2*x(1)-5;
g(2)= 2*x(2)-5;
g(3)= 4*x(3)-21;
g(4)= 2*x(4)+7; 

h = diag([2,2,4,2]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cineq, Gcineq, Hcineq]= confun(x)
% Constraint function
cineq = zeros(3,1);
cineq(1) = 8 - x(1)^2 -  x(2)^2-x(3)^2 - x(4)^2 - x(1) + x(2) - x(3) + x(4); 
cineq(2) = 10- x(1)^2 -2*x(2)^2-x(3)^2 - 2*x(4)^2 +   x(1)  + x(4)         ;
cineq(3) = 5-2*x(1)^2 -  x(2)^2-x(3)^2          - 2*x(1) + x(2)      + x(4);  

% Gradients of the constraint functions wrt. x
Gcineq=[-2*x(1)-1, -2*x(2)+1, -2*x(3)-1, -2*x(4)+1; 
       -2*x(1)+1, -4*x(2),   -2*x(3),   -4*x(4)+1;
       -4*x(1)-2, -2*x(2)+1, -2*x(3),   1];
Gcineq = Gcineq'; 

Hcineq = cell(1, length(cineq)); 
Hcineq{1} = diag([-2, -2, -2, -2]);
Hcineq{2} = diag([-2, -4, -2, -4]);
Hcineq{3} = diag([-4, -2, -2,  0]);

% % if fmincon, or the interior points on paper:   c<0
cineq = -cineq;
Gcineq = -Gcineq; 
Hcineq{1} = -1.*Hcineq{1}; 
Hcineq{2} = -1.*Hcineq{2};
Hcineq{3} = -1.*Hcineq{3};
end