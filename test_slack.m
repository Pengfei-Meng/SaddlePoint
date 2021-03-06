function test_slack()

% study incorporating bound constraints to Newton-Krylov solves

n = 3;    % design
m = 6;    % lam, slack
mu = 0.5; 

x = [5,5,5]';   lam = -1.*ones(6,1);  
[c_bnd, Gc_bnd]= coninequ(x); 
S = c_bnd;                   
% S = ones(6,1); 
[f,g,h] = objfun(x);

for k = 1:5
    % note S cannot have zero entries
    kkt_matrix = [h,           zeros(n, m),  Gc_bnd'; 
                  zeros(m,n),  diag(lam./S), eye(m); 
                  Gc_bnd,      eye(m),      zeros(m)]; 

    kkt_rhs = -[g' + lam(1:m/2) - lam(m/2+1:end);
                -mu.*(1./S) - lam; 
                c_bnd - S];

    dx = gmres(kkt_matrix, kkt_rhs);       

    x = x + dx(1:n); 
    S = S + dx(n+1 : n+m); 
    lam = lam + dx(n+m+1 : end); 
    
    [c_bnd, Gc_bnd]= coninequ(x); 
    [f,g,h] = objfun(x);
end



end

function [f,g,h] = objfun(x)
%     minimize    1000 - x1^2 - 2*x2^2 - x3^2 - x1*x2 - x1*x3
%     subject to  8*x1 + 14*x2 + 7*x3         = 56
%                 (x1^2 + x2^2 + x3^2 - 25) >= 0
%                 0 <= (x1, x2, x3) <= 10
%   and has two local solutions:
%   the point (0,0,8) with objective 936.0, and
%   the point (7,0,0) with objective 951.0    
% Objective function
f = 1000 - x(1)^2 - 2*x(2)^2 - x(3)^2 - x(1)*x(2) - x(1)*x(3);

% its derivative wrt. x
g(1)=-2*x(1)-x(2)-x(3);
g(2)=-4*x(2)-x(1);
g(3)=-2*x(3)-x(1);

h = [-2, -1, -1;
     -1, -4,  0;
     -1,  0, -2]; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ceq,Gceq]= conequ(x)
% Constraint function

ceq= 8*x(1) + 14*x(2) + 7*x(3) - 56; 

% Gradients of the constraint functions wrt. x
if nargout > 2
   Gceq(1,1)=8.;
   Gceq(2,1)=14.;
   Gceq(3,1)=7.;
end
end


function [c_bnd,Gc_bnd]= coninequ(x)
% Constraint function

c_nl = (x(1)^2 + x(2)^2 + x(3)^2 - 25);

% Gradients of the constraint functions wrt. x
if nargout > 2
   Gc_nl = [2*x(1), 2*x(2), 2*x(3)];
end

% 0 <= (x1, x2, x3) <= 10
c_bnd = zeros(6,1); 
c_bnd(1) = x(1);
c_bnd(2) = x(2);
c_bnd(3) = x(3);
c_bnd(4) = 10-x(1);
c_bnd(5) = 10-x(2);
c_bnd(6) = 10-x(3);
Gc_bnd = [eye(3);
         -eye(3)];
end

