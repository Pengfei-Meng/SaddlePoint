function main_interior()
% min  f(x)              min  f(x) - (1-mu) ln(S)
% s.t. h(x) = 0          s.t. h(x) = 0
%      g(x) >= 0              g(x) - s = 0

% L(x) = f(x) - (1-mu) ln(S) + lam_h h(x) + lam_g (g(x)-s)
% mu : 0 --> 1

mu = 0; 
epsilon_mu = 1e-2;     epsilon_tol = 1e-6; 
theta = 0.5; 

x = [2;2;2];
s = ones(8,1); 
[f,g] = objfun(x); 
[c,ceq,Gc,Gceq]= constfun(x); 



end







function [f,g] = objfun(x)
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
if nargout > 1
   g(1)=-2*x(1)-x(2)-x(3);
   g(2)=-4*x(2)-x(1);
   g(3)=-2*x(3)-x(1);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c,ceq,Gc,Gceq]= constfun(x)
% Constraint function
ceq=[];
c(1)= (x(1)^2 + x(2)^2 + x(3)^2 - 25);

% Gradients of the constraint functions wrt. x
if nargout > 2
   Gc(1,1)=2*x(1);
   Gc(2,1)=2*x(2);
   Gc(3,1)=2*x(3);
   Gceq=[];
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [H]= hessfun(x,lambda)
% Hessian function.
% Only need to specify structural non-zero elements of the upper
% triangle (including diagonal)  
  
H(1,1) = -2 + lambda.ineqnonlin*(-2);
H(1,2) = -1;
%H(2,1) = H(1,2);
H(1,3) = -1;
%H(3,1) = H(1,3);
H(2,2) = -4 + lambda.ineqnonlin*(-2);
H(3,3) = -2 + lambda.ineqnonlin*(-2);
H = sparse(H);
end