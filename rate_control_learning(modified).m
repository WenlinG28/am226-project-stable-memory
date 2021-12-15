%%%%
%%%%  modified from https://github.com/lsusman/stable-memory/blob/master/rate_control_learning.m

%  Simulate rate network with rate-control homeostasis
%%%%  Dynamically learn one memory item

clear all; clc;
close all;

load('data'); 
rng(data.rcseed);

N = 128;
dt = .1;
etaS = .01;
g = 1.0;
tau = 20;
tauy = 100;
Nmem = 1;

TotalSteps = 3000/dt - 1;
CalcEvery = 10/dt;
Nsteps = (TotalSteps+1)/CalcEvery;

x = randn(N,1);
x_all = nan(N,TotalSteps);
xlp = randn;
W_all = nan(N,N,Nsteps);
hs_all = nan(N,N,Nsteps);
noise_all = nan(N,N,Nsteps);
W = 2*randn(N)/sqrt(N);
r0 = 2*(rand(N,1) - .5);
zeta = .01;
y = randn(N,1);

H = sign(randn(N,N)/sqrt(N));
u = H(1:Nmem,:)'/sqrt(N);
v = H(Nmem+1:2*Nmem,:)'/sqrt(N);
input1 = zeros(N,TotalSteps);
input2 = zeros(N,TotalSteps);
base = 500/dt;
inLen = 100/dt;
input = 0;
overlaps = zeros(Nsteps,1);

f = waitbar(0,'Simulating, please wait...');

%%%% Construct input signal for learning
for i = 1:Nmem
    input1(:,base+1:base+inLen) = repmat(u(:,i),1,inLen);
    input2(:,base+1:base+inLen) = repmat(v(:,i),1,inLen);
end

%%%% Evolve network
for i=1:TotalSteps
   
    x_all(:,i) = x;
    r = tanh(g*x);
    xlp = ((-xlp + x/5e-2)/tau)*dt;
    y = y + (r - y)*dt/tauy; 
    input = input + (-zeta*input + randn*input1(:,i) + randn*input2(:,i))*dt;
    
    x = x + (-x + W*r + 10*input)*dt;
    L = r*y' - y*r';
    hs = (r0 - r)*r'*W;

    noise = (1*randn(N,N))/sqrt(N);
    W = W + etaS*(noise + hs + L)*dt;
    
    if ~mod(i-1,CalcEvery)
        W_all(:,:,(i-1)/CalcEvery + 1) = W;
        [V,D] = eig(W);
        [~,I] = sort(imag(diag(D)),'descend');
        v1 = u; u1 = v;
        v2 = real(V(:,I(1)));
        u2 = imag(V(:,I(1)));

        r_u1 = sqrt(power(u1'*u2,2)+power(u1'*v2,2));
        r_v1 = sqrt(power(v1'*u2,2)+power(v1'*v2,2));
        overlaps((i-1)/CalcEvery + 1) = sqrt(r_u1*r_u1+r_v1*r_v1);
        waitbar((i-1)/TotalSteps,f,'Simulating, please wait...');
    end
end

%%%% Compute eigenspectrum of W over time
[Vseq,Dseq] = eigenshuffle(W_all);
[~,I] = sort((imag(Dseq(:,end))),'descend');
Dsorted = Dseq(I,end);
taxis = [1:1:Nsteps]*CalcEvery*dt;

%%%% Plot spectrum
for j=1:1:N
    figure(100); subplot(2,1,1); plot(taxis,real(Dseq(I(j),:)),'linewidth',2,'color',[1 1 1]*.8); hold on;
    figure(100); subplot(2,1,2); plot(taxis,imag(Dseq(I(j),:)),'linewidth',2,'color',[1 1 1]*.8); hold on;
end

subplot(2,1,2); plot(taxis,imag(Dseq(I(1),:)),'linewidth',2,'color',[0 0.4470 0.7410]); hold on;
plot(taxis,imag(Dseq(I(end),:)),'linewidth',2,'color',[0 0.4470 0.7410]);
%%% -----new: plot shaded blue areas -----%%%
tinput = [base*dt, base*dt+inLen*dt];
minmax = [-10,20; -10,20];
shaded = area(tinput,minmax,'FaceColor',[0 0.4470 0.7410],'EdgeColor',[0 0.4470 0.7410]);
alpha(shaded, .2);
% %%% -----new: plot shaded red areas -----%%%
tinput = [900,1000];
shaded = area(tinput,minmax,'FaceColor',[0.8500 0.3250 0.0980],'EdgeColor',[0.8500 0.3250 0.0980]);
alpha(shaded, .2);
%%% -------------------------------- %%%

xlabel('time'); ylabel('Im(\lambda)'); set(gca,'fontsize',14); 

subplot(2,1,1); plot(taxis,real(Dseq(I(1),:)),'linewidth',2,'color',[0 0.4470 0.7410]); hold on;
plot(taxis,real(Dseq(I(end),:)),'linewidth',2,'color',[0 0.4470 0.7410]); 
%%% -----new: plot shaded blue areas -----%%%
tinput = [base*dt, base*dt+inLen*dt];
minmax = [-20,40; -20,40];
shaded = area(tinput,minmax,'FaceColor',[0 0.4470 0.7410],'EdgeColor',[0 0.4470 0.7410]);
alpha(shaded, .2);
%%% -----new: plot shaded red areas -----%%%
tinput = [900,1000];
shaded = area(tinput,minmax,'FaceColor',[0.8500 0.3250 0.0980],'EdgeColor',[0.8500 0.3250 0.0980]);
alpha(shaded, .2);
%%% -------------------------------- %%%
ylabel('Re(\lambda)');set(gca,'fontsize',14);

%%% -----new: plot overlaps -----%%%
figure(101); 
plot(taxis,overlaps,'linewidth',2,'color',[0 0.4470 0.7410]); hold on;
tinput = [base*dt, base*dt+inLen*dt];
minmax = [0,1.2; 0,1.2];
shaded = area(tinput,minmax,'FaceColor',[0 0.4470 0.7410],'EdgeColor',[0 0.4470 0.7410]);
alpha(shaded, .2);

tinput = [900,1000];
shaded = area(tinput,minmax,'FaceColor',[0.8500 0.3250 0.0980],'EdgeColor',[0.8500 0.3250 0.0980]);
alpha(shaded, .2);
xlabel('time'); ylabel('Overlaps');
%%% -------------------------------- %%%

close(f);