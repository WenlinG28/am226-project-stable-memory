
% Simulate erosion of real-coded memories over time
% in three different homeostatic mechanisms
% using different sigmoid functions

clear all; clc;
close all;

%% constant
% plasticity rate
eta = 0.01; 
% number of neurons
N = 128; 
% for dissipative synaptic dynamics
beta = 0.1; 
% timescale related to first-order low-passed version of x, x_bar
% used in decorrelation case
tau_x = 20; 
% rate control target rate
% independently drawed from a uniform distribution over the interval [−1, 1]
phi0 = 2*(rand(N,1) - .5); 

%% time-related parameters
% time constant
dt = 0.1; 
% total duration 10,000 as in the paper
TotalSteps = 10000/dt - 1;
% calculate 100 times
CalcEvery = 100/dt;
% number of steps to stimulate
Nsteps = (TotalSteps+1)/CalcEvery;

% for collection purpose
W_all = nan(N,N,Nsteps);

%% variants
% input x
x = randn(N,1);
% a first-order low-passed version of x
% phi_post(x) = phi(x - x_bar)
% see 'methods' in the original paper
x_bar = randn;
% connectivity matrix
W = zeros(N);
% W = 3 * randn(N)/sqrt(N);

%% simulation
for i=1:TotalSteps
%%% choose one sigmoid and comment the others
%     % sigmoid function 1
    phi = tanh(x); 
    phi_post = tanh(x - x_bar);
%     % sigmoid function 2
%     phi = max(-5,x);
%     phi_post = max(-5, x - x_bar);
%     % sigmoid function 3
%     phi = 1/(1+exp(-x)); phi = phi';
%     phi_post = 1/(1+exp(-x+x_bar)); phi_post = phi_post';

    x_bar = ((-x_bar + x/5e-2)/tau_x)*dt;
    x = x + (-x + W*phi)*dt; % note: no external input
    
%%% choose one homeostatic mechanism and comment the others
%     hm = -beta*W; % dissipation
    hm = (phi0 - phi)*phi'*W; % rate_control
%     hm = 0.5 * eye(N) - phi_post*phi'; % decorrelation

    noise = (1*randn(N,N))/sqrt(N);
    W = W + eta*(noise + hm)*dt;

    if ~mod(i-1,CalcEvery)
        W_all(:,:,(i-1)/CalcEvery + 1) = W;
    end

    if i == 2500/dt
        u = randn(N,1)/sqrt(N);
        v = randn(N,1)/sqrt(N);
        % real coding
        W = W + 5*(u*u');
    end

end

%% calcaulte eigenvalues
[Vseq,Dseq] = eigenshuffle(W_all);
[~,I] = sort((real(Dseq(:,26))),'descend');

%% plotting
taxis = [1:1:Nsteps]*CalcEvery*dt;
for j=1:1:N 
    plot(taxis,real(Dseq(j,:)),'linewidth',2,'color',[1 1 1]*.8); hold on;  
end
plot(taxis,real(Dseq(I(1),:)),'linewidth',2,'color',[0 0.4470 0.7410]); 
% ylim([-5 6]); 
set(gca,'fontsize',18);ylabel('Re(\lambda)');
xlabel('time'); box off;