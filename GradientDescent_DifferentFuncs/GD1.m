clear all; %remove all the old variables in the workspace
close all;

thetas = [0;0];
Js = [];
theta1 = [];
theta2 = [];
iteration = 0;
eta = 0.01;
counter = 0;
while (iteration < 1000)
    
    iteration = iteration + 1;
    thetas(1,1) = thetas(1,1) - ((eta)*(2*((thetas(1,1) + thetas(2,1) - 2))));
    thetas(2,1) = thetas(2,1) - ((eta)*(2*((thetas(1,1) + thetas(2,1) - 2))));%- ((eta)*(thetas(2,1)));
    %thetas = thetas - ((eta*(sum(thetas) - 2.0))*(ones(2,1))) - ((eta)*(thetas));
    J = (thetas(1,1) + thetas(2,1) - 2.0)^2;
    Js(iteration, 1) = J;
    theta1(iteration, 1) = thetas(1,1);
    theta2(iteration, 1) = thetas(2,1);
end

plot3(theta1, theta2,Js);
title('Gradient Descent');
xlabel('theta1');
ylabel('theta2');
zlabel('J');
savefig('GD1.fig');