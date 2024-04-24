%% 
% SUMMARY: compares rate of convergence for computing OT for a 
% fixed run time.

clear all

%% Experiment parameters
tau=0.5;    %0.5/0.2
eta = 50;    %50/130            % Greenkhorn's regularization parameter
full_iters = 5;
greedy_downsampling = 40;
small_iters = 28*28*full_iters;  % number of updates in experiment
use_synth_imgs = true;

%% Create input
addpath(genpath('input_generation/'));
addpath(genpath('input_generation/mnist'));
m=28; % images are of dim mxm
n=m*m;

if use_synth_imgs
    fraction_fg = 0.2; % parameter: 20% of image area will be foreground
    img_1 = synthetic_img_input(m, fraction_fg);
    img_2 = synthetic_img_input(m, fraction_fg);
    flattened_img_1 = reshape(img_1,n,1);
    flattened_img_2 = reshape(img_2,n,1)';
else
% MNIST IMAGE INPUT
    imgs = loadMNISTImages('t10k-images-idx3-ubyte');
    labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    
    % choose random pair of mnist images
    nimages = size(imgs,2);
    idx_1 = randi(nimages);
    idx_2 = randi(nimages);
    if idx_2 == idx_1 %% ensure distinctness
        idx_2 = idx_1 + 1;
    end
    % flatten images and add small background so nonzero entries
    flattened_img_1 = imgs(:,idx_1) +0.01*ones(n,1);
    flattened_img_2 = imgs(:,idx_2)'+0.01*ones(1,n);
end    

[A,r,c,C] = ot_input_between_imgs(flattened_img_1,flattened_img_2,eta,m,n);
% size(r)
% Initialize at same point as GCPB for apples-to-apples comparison
u = zeros(n,1);
for i=1:n
    temp_sum = 0;
    for l=1:n
        temp_sum = temp_sum + (c(l) * exp( (-1* C(i,l))*eta ));
    end
    u(i) = -1 * log(temp_sum)/eta;
end

% Extract A
%A = zeros(n,n);
%for i=1:n
%    for j=1:n
%        A(i,j) = r(i)*c(j)*exp( eta*(u(i) + - C(i,j)) );
%    end
%end

% Desired time
time = 15;

%% Run algorithms
% Run Sinkhorn
disp('starting Sinkhorn');
%[sink_iter,sink_time, sink_ots] = Sinkhorn(r,c,0.05,true); 
[P_sink, err_sink, sink_ots,iter_sink] = sinkhorn_tm(A,r,c,true,C,time);
sink_ots = nonzeros(sink_ots);

% Run APDAGD Algorithm
[iter_agd,agd_ots] = APDAGD_tm(A,r,c',C,0.05,true,time);
agd_ots = nonzeros(agd_ots);

% Run Greenkhorn algorithm
addpath(genpath('algorithms/'));
disp('Beginning to run Greenkhorn.')
compute_otvals = true;
[P_greedy,err_greedy,greedy_ot,iter_greedy] = greenkhorn_tm(A,r,c,compute_otvals,C,time);
greedy_ots = greedy_ot';
greedy_ots = nonzeros(greedy_ots);

% Run GCPB algorithm
% disp('Beginning to run GCPB.')
% compute stepsizes
%r_max = max(r);
%eps = 1/eta;
%L = r_max/eps;
% stepsizes = [1/(L*n); 3/(L*n); 5/(L*n)];
%stepsizes = 3/(L*n);

%runs=1;
%runs = size(stepsizes,1);
%gcpb_ots = [];
%for run=1:runs
%    disp([' --> Beginning run ',num2str(run),' of ',num2str(runs)])
%    stepsize = stepsizes(run);
%    [gcpb_ot_output,iter_gcpg] = gcpb_ot_tm(r,c,C,eps,stepsize,time);
    % g3pb_ots = [gcpb_ots; gcpb_ot_output'];
%    gcpb_ots = gcpb_ot_output';
%    gcpb_ots = nonzeros(gcpb_ots);
%end

% Compute gold standard: linear program to solve OT
disp('Computing gold standard');
lp_opt = computeot_lp(C,r,c,n);


%% Make plot
downsample_indices        = linspace(1,small_iters+1,full_iters+1);
greedy_downsample_indices = linspace(1,small_iters+1,1+small_iters/greedy_downsampling);
iter_max = max([length(greedy_ots), iter_sink, iter_agd]);
x_range = linspace(0, iter_max, 1000);

% Load MIT colors
mit_red    = [163, 31, 52]/255;
mit_grey   = [138, 139, 140]/255;
mit_green = [38, 158, 78]/255;
mit_blue = [48, 103, 186]/255;

% Plot gold standard (LP)
lp_opt_vec = lp_opt*ones(1,time*10);
plot(1:time*10, lp_opt_vec,'DisplayName','True optimum','Color','black','LineWidth',2); 
hold('all')

% Plot GREENKHORN
linestyle='-';
plot(linspace(1,time*10,length(greedy_ots)),greedy_ots,'DisplayName','GREENKHORN','Color',mit_red,'LineStyle',linestyle,'LineWidth',2);
hold('all')

% Plot Sinkhorn
plot(linspace(1,time*10,length(sink_ots)),sink_ots,'DisplayName','Sinkhorn','Color',mit_blue,'LineStyle',linestyle,'LineWidth',2);
hold('all')

% Plot AGD
plot(linspace(1,time*10,length(agd_ots)),agd_ots,'DisplayName','AGD','Color',mit_green,'LineStyle',linestyle,'LineWidth',2);
% plot(1:(small_iters+1), agd_otval,'DisplayName','AGD','Color',mit_green,'LineStyle',linestyle,'LineWidth',2);
hold('all')

% Plot GBCP2
%linestyle='--';
%plot(linspace(1,time*10,length(gcpb_ots)),gcpb_ots,'DisplayName',['SAG, stepsize=3/(Ln)'],'Color',mit_grey,'LineStyle',linestyle,'LineWidth',2);


%for run=1:runs
%    stepsize = stepsizes(run);
%    if run==1
%        linestyle='-.';
%        stepsizestr='1';
%    elseif run==2
%        linestyle='--';
%        stepsizestr='3';
%    else
%        linestyle=':';
%        stepsizestr='5';
%    end 
%    plot(greedy_downsample_indices, gcpb_ots(run,greedy_downsample_indices),'DisplayName',['SAG, stepsize=',num2str(stepsizestr),'/(Ln)'],'Color',mit_grey,'LineStyle',linestyle,'LineWidth',2);
%end

hold('all')
hold('off');
legend('show');
ylabel('Value of OT');
xlabel('time unit: 0.1s');
title(strcat('Sinkhorn vs GREENKHORN vs AGD for OT, tau=',num2str(tau)));

