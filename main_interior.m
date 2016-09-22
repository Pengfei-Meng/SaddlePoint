function main_interior()
% min  f(x)              min  f(x) - mu ln(S)
% s.t. h(x) = 0          s.t. h(x) = 0
%      g(x) >= 0              g(x) - s = 0
% As in the paper, f, h, g all should be second order differentiable
% no bound constraint

% L(x) = f(x) - (1-mu) ln(S) + lam_h h(x) + lam_g (g(x)-s)
% mu : 1 --> 0

mu = 1; 
epsilon_mu = 1e-1;     epsilon_tol = 1e-6; 
theta = 0.5; 

n = 4;    m = 3; 
x = [1;1;1;1];     lam = ones(3,1);
S = [4; 6; 1]; 

[fobj,gobj,hobj] = objfun(x); 
[Cg, ~, Ag, ~, Hg]= coninequ(x);

E_mu0 = E_inf(0.0);
E_local = E_mu0;

glob = true; 

while E_mu0 > epsilon_tol
    
  % Algorithm II inner loop, solve for one value of mu
  while E_local > epsilon_mu
      
      % how to add the trust radius into the kkt_solve? 
      [dx] = kkt_matrix(x, S, lam, mu, gobj,hobj,Cg,Ag,Hg); 

      if glob
          x_temp = x + dx(1:n);
          S_temp = S + dx(n+1:n+m);
          lam_temp = lam + dx(n+m+1:end); 
          
          nu = 1.0;
          actual_red = merit_phi(x,S,nu,mu) - merit_phi(x_temp,S_temp,nu,mu);
          % predit_red = -q_tang_red() + nu*v_pred()
          
      else
          x = x + dx(1:n);
          S = S + dx(n+1:n+m);
          lam = lam + dx(n+m+1:end); 
      end
      
      [fobj,gobj,hobj] = objfun(x); 
      [Cg, ~, Ag, ~, Hg]= coninequ(x);
      E_local = E_inf(mu);    % E_local < epsilon_mu  %(it has to be)

  end
  
  E_mu0 = E_inf(0.0);
    
  mu = theta*mu;
  epsilon_mu = theta*epsilon_mu; 
  
  if any(S<0)
      S           % strange that S is indeed positive all the time
      pause
  end
  
  if any(lam<0)
      lam
      pause
  end
 
end

% options = optimoptions(@fmincon,'Algorithm','interior-point',...
%     'GradObj','on',...
%     'GradConstr','on','DerivativeCheck','on'); 
% [x,f] = fmincon(@objfun, x, [],[],[],[],[],[],@coninequ, options)

    function E_ = E_inf(mu_local)
        E = [gobj + Ag*lam;
            S.*lam - mu_local.*ones(size(S));
            Cg + S];
        E_ = norm(E, Inf);
    end

end

function phi = merit_phi(x,S,nu,mu)

  if any(S<0)
      sprintf('Negative Slack variable in merit_phi func')  
      pause
  end

    [fobj,gobj,hobj] = objfun(x); 
    [Cg, ~, Ag, ~, Hg]= coninequ(x);
    
    % note, here inequality constraints only! 
    phi = fobj - mu*sum(log(S)) + nu * norm(Cg + S,2);

end


function [dx] = kkt_matrix(x, S, lam, mu, gobj,hobj,Cg,Ag,Hg)

    n = length(x);
    m = length(S);
    
%     [fobj,gobj,hobj] = objfun(x); 
%     [Cg, ~, Ag, ~, Hg]= coninequ(x);
    hlag = hobj+lam(1).*Hg{1} + lam(2).*Hg{2} + lam(3).*Hg{3}; 
    
    kkt_mat = [hlag,        zeros(n, m),  Ag; 
               zeros(m,n),  diag(lam./S), eye(m); 
               Ag',      eye(m),      zeros(m)]; 
            
    kkt_rhs = -[gobj + Ag*lam;
                -mu.*(1./S) + lam; 
                Cg + S];
            
%     E = [gobj + Ag*lam;
%         S.*lam - mu.*ones(size(S));
%         Cg + S];
%     E_inf = norm(E, Inf);
            
    [dx,flag,relres] = gmres(kkt_mat, kkt_rhs);       
    % relres
    % c_s = cineq + S        % cineq + S = 0 is maintained towards solution
    % norm(kkt_rhs, Inf)
    % cond(kkt_mat)
    
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
