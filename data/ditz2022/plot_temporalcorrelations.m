%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Temporal Correlations %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script plots
% 1) tuning curves of three exemplary temporally correlated cell pairs.
% 2) the respective cross correlograms with significant troughs/peaks.

clear, clc
close all

load('data_tempcorr.mat')

%% plot figure 7

% the exemplary neuron pairs:
% 60,61 (data set a), 4 in cc_timing
% 407,408 (data set a), 12 in cc_timing
% 178,179 (data set b), 19 in cc_timing

% set the parameters
data_table = data_table_b;
neuron1 = 178;
neuron2 = 179;
idx_pair = 19; % index for cc_timing and threshold
max_num = 7; %5 %7
ticks = [1 2 4 7 12 20 30];
%[1 2 3 4 5];
%[1 2 4 7 12 20 30];

% first plot the tuning curves
tuning_1 = data_table{2,neuron1};
tuning_2 = data_table{2,neuron2};

% cell class of the neurons
class_1 = data_table{1,neuron1}
class_2 = data_table{1,neuron2}
% for NSBS pairs plot BS (1) neurons always on the left and NS (2) neurons 
% always on the right

figure(1)
% first tuning curve
yyaxis left % BS, 1
plot(tuning_1)
ylabel('Firing rate (Hz)')
set(gca,'XTick',1:max_num);
set(gca,'XTickLabel',ticks)
xlim([0.9 max_num+0.1])
xlabel('Numerosity')

% second tuning curve
yyaxis right % NS, 2
plot(tuning_2)
ylabel('Firing rate (Hz)')

% plot the cc_timing value: corrected CCH, small
CCH_corrected = cc_timing{idx_pair};

figure(2)
bar(CCH_corrected, 'BarWidth', 1, 'FaceColor',[0 0 0])
hold on
yline(median(CCH_corrected)+threshold(idx_pair)) % peak has to be larger than this line
hold on
yline(median(CCH_corrected)-threshold(idx_pair)) % peak has to be larger than this line
hold off
xlabel('Time delay (ms)');
ylabel('Normed spike count');
set(gca,'XTick',1:50:101); % for 1ms
set(gca,'XTickLabel',-50:50:50);
xlim([1 101])