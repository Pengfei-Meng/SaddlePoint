function main_interior()
% min  f(x)              min  f(x) - mu ln(S)
% s.t. h(x) = 0          s.t. h(x) = 0
%      g(x) >= 0              g(x) - s = 0
% As in the paper, f, h, g all should be second order differentiable
% no bound constraint

% L(x) = f(x) - (1-mu) ln(S) + lam_h h(x) + lam_g (g(x)-s)
% mu : 1 --> 0

mu = 1; 
epsilon_mu = 1e-2;     epsilon_tol = 1e-6; 
theta = 0.5; 

n = 4;    m = 3; 
x = [1;1;1;1];     lam = ones(3,1);
s = [4; 6; 1]; 

for k = 1:80
  dx = kkt_matrix(x, s, lam, mu); 
  x = x + dx(1:n);
  s = s + dx(n+1:n+m);
  lam = lam + dx(n+m+1:end); 
  if any(s<0)
      s
      pause
  end
end
x

% options = optimoptions(@fmincon,'Algorithm','interior-point',...
%     'GradObj','on',...
%     'GradConstr','on','DerivativeCheck','on'); 
% [x,f] = fmincon(@objfun, x, [],[],[],[],[],[],@coninequ, options)

end


function dx = kkt_matrix(x, S, lam, mu)

    n = length(x);
    m = length(S);
    
    [fobj,gobj,hobj] = objfun(x); 
    [cineq, ~, Gcineq, ~, hcineq]= coninequ(x);
    hlag = hobj+lam(1).*hcineq{1} + lam(2).*hcineq{2} + lam(3).*hcineq{3}; 
    
    kkt_mat = [hlag,        zeros(n, m),  Gcineq; 
               zeros(m,n),  diag(lam./S), eye(m); 
               Gcineq',      eye(m),      zeros(m)]; 
            
    kkt_rhs = -[gobj + Gcineq*lam;
                -mu.*(1./S) + lam; 
                cineq + S];
            
    dx = gmres(kkt_mat, kkt_rhs);       

    c_s = cineq + S
    norm(kkt_rhs, Inf)
    norm(kkt_mat)
    
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

function [cineq, ceq, Gcineq, Gceq, Hcineq]= coninequ(x)
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
ceq = [];  Gceq = []; 

Hcineq = cell(1, length(cineq)); 
Hcineq{1} = diag([-2, -2, -2, -2]);
Hcineq{2} = diag([-2, -4, -2, -4]);
Hcineq{3} = diag([-4, -2, -2,  0]);

% % if fmincon, or the interior points on paper:   c<0
cineq = -cineq;
Gcineq = -Gcineq; 
Hcineq{1} = -1*Hcineq{1}; 
Hcineq{2} = -1*Hcineq{2};
Hcineq{3} = -1*Hcineq{3};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
