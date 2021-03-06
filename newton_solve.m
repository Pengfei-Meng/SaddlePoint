function newton_solve()
%--------------------------------------------------------------------------
% Example nonlinear system of equations formulated as a Matlab model used
% to demonstate using the Knitro Mex interface with Matlab.
% The problem is
%
%                      sin(x1) - x2^2 = 0
%               cos(x1) + x1 + x3 - 2 = 0
%                     x4^2 + x3^2 - 7 = 0
%     x5*x4^2*5/2 + x5 - 10*x4^2 +  4 = 0
%     x5*x4^2*5/2 - x5 - 10*x4^2 + 11 = 0
%               (x6+x7)/(x6^2) - x4^2 = 0
%                    x6 + x7 - x2 - 8 = 0
%
% One solution is approximately x* = [0;0;1;2.45;3.5;1.15;6.85]
%--------------------------------------------------------------------------
addpath('../../MATLAB/knitromatlab')
options = optimset('Display','iter','Algorithm','active-set');
%options = optimset('DerivativeCheck','on','Jacobian','on','Display','iter','Algorithm','sqp','FinDiffType','central');
%options = optimset('DerivativeCheck','on','Jacobian','on','Display','iter','Algorithm','trust-region-dogleg');
%options = optimset('DerivativeCheck','on','Jacobian','on','Display','iter','Algorithm','levenberg-marquardt');
%options = optimset('DerivativeCheck','on','Jacobian','on','Display','iter','Algorithm','trust-region-reflective');

% [x,fval] = knitromatlab_fsolve(@myfun, ones(7,1), [], options)
% [x_fsolve, fval_fsolve] = fsolve(@myfun, ones(7,1))
figure()
x_star = [0;0;1;2.45;3.5;1.15;6.85]; 

[x_newton, f_newton, x_hist, dx_hist] = newton(@myfun, 2.*ones(7,1)); 

radius = 2;
[x, FVAL, Xhist, DxHist] = trunc_newton(@myfun, 2.*ones(7,1), radius); 

end


function [x, FVAL, x_hist, dx_hist] = trunc_newton(FUN, x, radius)

[f, g] = FUN(x); 
f_hist = []; 
f_hist = [f_hist, norm(f)]; 
x_hist = []; 
dx_hist = [];

x_hist = [x_hist, x]; 

iter = 0; 
while norm(f, Inf) > 1e-5 
    iter = iter + 1; 
    
    condest(g);
    dx = - g\f;
    
    % trucate dx, || dx || <= radius
    ind = abs(dx) > radius; 
    dx(ind) = radius*( sign(dx(ind) ) ); 
    dx_truced = dx ;
    x = x + dx;  
    
    x_hist = [x_hist, x]; 
    dx_hist = [dx_hist, dx]; 
    
    [f, g] = FUN(x); 
    f_hist = [f_hist, norm(f)]; 
end
FVAL = f; 
iter

hold on
semilogy(f_hist, 'b-o', 'LineWidth', 1)

end


function [x, FVAL, x_hist, dx_hist] = newton(FUN, x)
  
[f, g] = FUN(x); 
f_hist = []; 
f_hist = [f_hist, norm(f)];

x_hist = []; 
dx_hist = [];
x_hist = [x_hist, x]; 

iter = 0; 
while norm(f, Inf) > 1e-6  
    iter = iter + 1; 
    x;
    condest(g);
    dx = - g\f;
    x = x + dx;  
    [f, g] = FUN(x); 
    f_hist = [f_hist, norm(f)]; 
    x_hist = [x_hist, x]; 
    dx_hist = [dx_hist, dx]; 
end
FVAL = f; 
iter

semilogy(f_hist, 'r-o', 'LineWidth', 1)
hold on
end


function [F,g] = myfun(x)
F = [sin(x(1))-x(2)^2; ...
     cos(x(1))+x(1)+x(3)-2; ...
     x(4)^2+x(3)^2-7; ...
     x(5)*x(4)^2*5/2+x(5)-10*x(4)^2+4; ...
     x(5)*x(4)^2*5/2-x(5)-10*x(4)^2+11; ...
     (x(6)+x(7))/(x(6)^2)-x(4)^2; ...
     x(6)+x(7)-x(2)-8];
if nargout > 1
    g = sparse(7);
    g(1,1) = cos(x(1));
    g(1,2) = -2*x(2);
    g(2,1) = -sin(x(1)) + 1;
    g(2,3) = 1;
    g(3,3) = 2*x(3);
    g(3,4) = 2*x(4);
    g(4,4) = 5*x(4)*x(5) - 20*x(4);
    g(4,5) = x(4)^2*5/2 + 1;
    g(5,4) = 5*x(4)*x(5) - 20*x(4);
    g(5,5) = x(4)^2*5/2 - 1;
    g(6,4) = -2*x(4);
    g(6,6) = (x(6)^2 - 2*x(6)*(x(6)+x(7))) / x(6)^4;
    g(6,7) = 1 / x(6)^2;
    g(7,2) = -1;
    g(7,6) = 1;
    g(7,7) = 1;
end
end