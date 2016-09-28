function main_interior2()
% Made on top of main_interior(), with equality constraints added
% min  f(x)              min  f(x) - mu ln(S)
% s.t. h(x) = 0          s.t. h(x) = 0
%      g(x) >= 0              g(x) - s = 0
% untested at the moment
% As in the paper, f, h, g all should be second order differentiable
% no bound constraint

% L(x) = f(x) - (1-mu) ln(S) + lam_h h(x) + lam_g (g(x)-s)
% mu : 1 --> 0

% ----- Matlab ----
% options = optimoptions(@fmincon,'Algorithm','interior-point',...
%     'GradObj','on',...
%     'GradConstr','on','DerivativeCheck','on'); 
% [x,f] = fmincon(@objfun, [-1,1], [],[],[],[],[],[],@coninequ, options)

mu = 1; 
epsilon_mu = 1e-1;     epsilon_tol = 1e-6; 
theta = 0.5; 

%----------------- problem dependent setting --------------%
% n = 4;    m = 3; 
% x = [1;1;1;1];     lam = ones(3,1);
% S = [4; 6; 1]; 
% % S = [1;1;1]; 

% note: only inequality has S; equality no S
% lam : has two components as well
n = 2;  
m_eq = 1;
m_ineq = 2; 
m = m_eq + m_ineq; 

x = ones(n,1);     
lam_eq = ones(m_eq,1);         % lam_eq  lam_ineq
lam_ineq = ones(m_ineq,1);
S = ones(m_ineq,1);       
%----------------------------------------------------------

[fobj,gobj,hobj] = objfun(x); 
[Cg, ceq, Ag, Ah, Hg, Hh]= coninequ(x);

E_mu0 = E_inf(0.0);
E_local = E_mu0;

glob = true; 
radius = 2.0;     % can be changed to other values, flexible with dim(x,s)
eta = 1e-8; 
tau = 0.995; 

while E_mu0 > epsilon_tol
  
  nu = 1.0; 
  % Algorithm II inner loop, solve for one value of mu
  while E_local > epsilon_mu
      % how to add the trust radius into the kkt_solve? 
      [dx] = kkt_matrix(x, S, lam_ineq, lam_eq, mu, gobj,hobj,Cg,Ag,Hg,ceq,Ah,Hh); 

      if glob
          % question remained
          % nu update? 
          % trust region with GMRES?
          % truncate dx respecting radius   
          %% trust region on Slack S
          dx_ = dx(1:n);
          ds_ = dx(n+1:n+m_ineq);
          dxs_scaled = norm([dx_;ds_./S],2);
          
          if dxs_scaled <= radius  
              sprintf('norm(dx, S^{-1}ds) <= radius')
          else
              sprintf('norm(dx, S^{-1}ds) > radius, being trucated...')
              ratio = radius/dxs_scaled; 
              dxs_scaled_mod = [dx_;ds_./S].*ratio; 
              dx_ = dxs_scaled_mod(1:n);
              ds_ = dxs_scaled_mod(n+1:n+m_ineq).*S; 
          end

          pos_vec = ds_ + tau.*S; 
          if all(pos_vec > 0)
              sprintf('new S positive')
          else
              ind = pos_vec < 0;   
              ds_(ind) = -tau.*S(ind); 
          end    
          
          %% Multipliers check
          if all( lam_ineq + dx(end-m_ineq+1:end) > 0 )
              sprintf('Multipliers tentative next all positive')
          else
              sprintf('Multipliers tentative next has negative entries')
              dx(end-m_ineq+1:end) = zeros(size(dx(end-m_ineq+1:end))); 
          end
                    
          x_temp = x + dx_;
          S_temp = S + ds_;
          % lam_temp_eq = lam + dx(n+m+1:end); 
          
          [predit_red,nu] = pred_red(x,S,mu,lam_eq,lam_ineq,dx,nu) ; 
          actual_red = merit_phi(x,S,nu,mu) - merit_phi(x_temp,S_temp,nu,mu);
          gamma = actual_red/predit_red; 
          
          if gamma >= eta
              x = x + dx_;
              S = S + ds_;
              lam_eq = lam_eq + dx(n+m_ineq+1:n+m_ineq+m_eq);  
              lam_ineq = lam_ineq + dx(end-m_ineq+1 : end);  
              
              if gamma >= 0.9
                  radius = max(7*norm(dx(1:n+m),2), radius);
              elseif gamma >= 0.3
                  radius = max(2*norm(dx(1:n+m),2), radius);
              else
                  radius = radius; 
              end
          else     % step is rejected
              radius = 0.3*radius; 
          end
          
          
      else
          x = x + dx(1:n);
          S = S + dx(n+1:n+m);
          lam_eq = lam_eq + dx(n+m_ineq+1:n+m_ineq+m_eq);  
          lam_ineq = lam_ineq + dx(end-m_ineq+1 : end);  
          
      end
      
      [fobj,gobj,hobj] = objfun(x); 
      [Cg, Ch, Ag, Ah, Hg, Hh]= coninequ(x);
      E_local = E_inf(mu);    % E_local < epsilon_mu  %(it has to be)
      
      x
      S
      % lam_ineq
      
  end
  
  E_mu0 = E_inf(0.0);
    
  mu = theta*mu;
  epsilon_mu = theta*epsilon_mu; 
  
  if any(S<0)
      S           % strange that S is indeed positive all the time
      sprintf('Negative Slack variable in this mu loop') 
      pause
  end
  
  if any(lam<0)
      lam
      pause
  end
 
end

x
fobj


    function E_ = E_inf(mu_local)                  % g: ineq   h: eq
        E = [gobj + Ag*lam_ineq + Ah*lam_eq;
            S.*lam_ineq - mu_local.*ones(size(S));
            ceq;
            Cg + S];
        E_ = norm(E, Inf);
    end

end


function [p_red,nu] = pred_red(x,S,mu,lam_eq,lam_ineq,dx,nu)
    [fobj,gobj,hobj] = objfun(x); 
    [Cg, Ch, Ag, Ah, Hg, Hh]= coninequ(x);

     n = length(x); 
     m_ineq = length(Cg);
     m_eq = length(Ch); 
     dx_ = dx(1:n);
     ds_ = dx(n+1:n+m_ineq);
     e = ones(length(S),1);
     
     hlag = hobj; 
     for j = 1:m_ineq
        hlag = hlag + lam_ineq(j).*Hg{j};
     end
    
     for j = 1:m_eq
        hlag = hlag + lam_eq.*Hh{j};
     end
          
     tan_red = gobj'*dx_ - mu*(e'*(ds_./S)) + ...
         0.5*(dx_'*hlag*dx_) + 0.5*(ds_'*(diag(lam_ineq./S))*ds_);
     
     gs = [Ch; Cg + S]; 
     ATv = [Ah'*dx_; Ag'*dx_ + ds_]; 
     vpred = norm(gs) - norm( gs + ATv ) ;  
     
     norm_obj = gs' * ATv + ATv'*ATv; 
     rho = 0.3; 
     
     if abs(norm_obj) < 1e-6
         nu = nu;
     else
         nu_lb = tan_red/((1-rho)*vpred); 
         nu = max(nu, nu_lb); 
     end
     p_red = -tan_red + nu*vpred;
       
end

function phi = merit_phi(x,S,nu,mu)

    [fobj,gobj,hobj] = objfun(x); 
    [Cg, Ch, Ag, Ah, Hg, Hh]= coninequ(x);
    
    % note, here inequality constraints only! 
    gs = [Ch; Cg + S]; 
    phi = fobj - mu*sum(log(S)) + nu * norm(gs,2);

end


function [dx] = kkt_matrix(x, S, lam_ineq, lam_eq, mu, gobj,hobj,Cg,Ag,Hg,ceq,Ah,Hh)

    n = length(x);
    m_eq = length(lam_eq);
    m_ineq = length(lam_ineq);
    
    hlag = hobj; 
     for j = 1:m_ineq
        hlag = hlag + lam_ineq(j).*Hg{j};
    end
    
    for j = 1:m_eq
        hlag = hlag + lam_eq.*Hh{j};
    end
        
    lam_S = lam_ineq./S;
    if any(lam_S<0)
        ind = lam_S < 0; 
        lam_S(ind) = mu./(S(ind).^2);         
    end
    sigma = diag(lam_S);
    
    kkt_mat = [hlag,        zeros(n, m_ineq),    Ah,              Ag; 
               zeros(m_ineq,n),  sigma,    zeros(m_ineq,m_eq),  eye(m_ineq); 
               Ah',        zeros(m_eq,m_ineq + m_eq + m_ineq);
               Ag',      eye(m_ineq),      zeros(m_ineq, m_eq + m_ineq)]; 
            
    kkt_rhs = -[gobj + Ag*lam_ineq + Ah*lam_eq;
                -mu.*(1./S) + lam_ineq;
                ceq; 
                Cg + S];                
                      
    [dx,flag,relres] = gmres(kkt_mat, kkt_rhs);       
    % relres
    % c_s = cineq + S        % cineq + S = 0 is maintained towards solution
    % norm(kkt_rhs, Inf)
    sprintf('Condition number of the kkt matrix') 
    cond(kkt_mat)
    
end

%{
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

function [cineq, ceq, Gcineq, Gceq, vargout]= coninequ(x)
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
vargout = Hcineq; 
end
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % Another nonlinear example
function [f,g,h] = objfun(x)
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

function [cineq, ceq, Gcineq, Gceq, Hcineq, Hceq] = coninequ(x)
% Nonlinear inequality constraints
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

% Nonlinear equality constraints
ceq = x(1)^2 + x(2)^2 - 4;
Gceq = [2*x(1);
        2*x(2)];   
Hceq = cell(1, length(ceq));    
Hceq{1} = [2,0;
           0,2]; 
end

