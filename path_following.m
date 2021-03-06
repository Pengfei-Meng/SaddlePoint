function path_following()
% Ignore this, this one is not working for some reason.
n = 2;
m_eq = 0;
m_ineq = 2;
m = m_eq + m_ineq;

x = ones(n,1);
lam_eq = ones(m_eq,1);         % lam_eq  lam_ineq
lam_ineq = ones(m_ineq,1);
%S = ones(m_ineq,1);
S = [0.5; 10];

[fobj,gobj,hobj] = objfun(x);
[Cg, ceq, Ag, Ah, Hg, Hh]= coninequ(x);


% --------- other parameters ---------
mu = 0.6;
epsilon_mu = 1e-1;     epsilon_tol = 1e-6;
theta = 0.5;

x0 = 1.2.*x; 

[E_mu0, kkt_rhs] = E_inf(0.0);
E_local = E_mu0;

outer_tol = E_mu0 * epsilon_tol;
inner_tol = E_mu0 * epsilon_mu;

%------------------------------------

while E_mu0 > outer_tol  %epsilon_tol  %outer_tol
    
    radius = 2.0;

    while E_local > inner_tol     %epsilon_mu
        
        % --------------predictor step----------------
        e = ones(n + m_ineq + m, 1);
        steep_descent = kkt_matrix_product(e);
        
        dx = -radius.*steep_descent;
               
        x = x + dx(1:n);
        S = S + dx(n+1:n+m_ineq);
        % lam_eq = lam_eq + dx(n+m_ineq+1:n+m_ineq+m_eq);
        lam_ineq = lam_ineq + dx(end-m_ineq+1 : end);
        
        % --------------correction step----------------
        [fobj,gobj,hobj] = objfun(x);
        [Cg, ceq, Ag, Ah, Hg, Hh]= coninequ(x);
        
        
        [dx,flag,relres,iter] = gmres(@kkt_matrix_product, kkt_rhs, [],[]);      
        
        x = x + dx(1:n);
        S = S + dx(n+1:n+m_ineq);
        % lam_eq = lam_eq + dx(n+m_ineq+1:n+m_ineq+m_eq);
        lam_ineq = lam_ineq + dx(end-m_ineq+1 : end);
        
        
        
        
    end
    
    
end



    function [E_, kkt_rhs] = E_inf(mu_local)
        %         E = [gobj + Ag*lam_ineq + Ah*lam_eq;
        %             S.*lam_ineq - mu_local.*ones(size(S));
        %             ceq;
        %             Cg + S];
        kkt_rhs = -[(1-mu).*gobj + mu.*(x-x0) + Ag*lam_ineq;
            S.*lam_ineq - mu_local.*ones(size(S));
            ceq;
            Cg + S];
        
        E_ = norm(kkt_rhs, Inf);
    end

    function Kv = kkt_matrix_product(in_vec)
        
        hlag = (1-mu).*hobj + mu.*eye(length(hobj));
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
        
%         kkt_mat = [hlag,        zeros(n, m_ineq),    Ah,              Ag;
%             zeros(m_ineq,n),  sigma,    zeros(m_ineq,m_eq),  eye(m_ineq);
%             Ah',        zeros(m_eq,m_ineq + m_eq + m_ineq);
%             Ag',      eye(m_ineq),      zeros(m_ineq, m_eq + m_ineq)];
        
        kkt_mat = [hlag,        zeros(n, m_ineq),               Ag;
            zeros(m_ineq,n),  sigma,      eye(m_ineq);
            Ag',      eye(m_ineq),      zeros(m_ineq,  m_ineq)];
        
        cond(kkt_mat)
        
        Kv = kkt_mat*in_vec;  
        Kv = Kv./norm(Kv); 
    end
end


function [dx] = kkt_matrix(x, S, lam_ineq, lam_eq, mu)

[fobj,gobj,hobj] = objfun(x);
[Cg, Ch, Ag, Ah, Hg, Hh]= coninequ(x);

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
ceq = [];
Gceq = [];
Hceq = [];
end