%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Tuning properties for adjacent neurons %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script plots
% 1) tuning curves of four exemplary NS-BS and four BS-BS neuron pairs.
% 2) histograms of CC_tuning values of numerosity encoding NS-BS and BS-BS
% cells.

clear, clc
close all

load('data_tuningadjacent.mat')

%% plot 6A and 6C

% set the parameters
correlation = correlation_a; %correlation_a %correlation_b %correlation_c
data_table = data_table_a; %data_table_a %data_table_b %data_table_c
neuron1 = 22; %dataset1: 22 43 61 477 415, dataset2: 267 291, dataset3: 240
neuron2 = 23; %dataset1: 23 42 60 479 416, dataset2: 268 290, dataset3: 241
max_num = 5; %5 %7 %4

% first plot the tuning curves
tuning_1 = data_table{2,neuron1};
tuning_2 = data_table{2,neuron2};

% cell class of the neurons
class_1 = data_table{1,neuron1}
class_2 = data_table{1,neuron2}
% for NS-BS pairs: plot BS (1) neurons always on the left and NS (2) 
% neurons always on the right

figure(1)
% first tuning curve
yyaxis left % BS, 1
plot(tuning_1)
ylabel('Firing rate (Hz)')
set(gca,'XTick',1:max_num);
xlim([0.9 max_num+0.1])
xlabel('Numerosity')

% second tuning curve
yyaxis right % NS, 2
plot(tuning_2)
ylabel('Firing rate (Hz)')

% then get the CC_Tuning
for i = 1:size(correlation,2)
    if correlation{1,i} == [neuron1,neuron2]
        tuning = correlation{2,i}
    end
end

%% plot 6B an 6D
% histograms for cell pairs
edges = [-1:0.5:1];

figure(2) % BSBS pairs
BB_histo = histogram(BSBS_pairs ,edges)
xlabel('Correlation Coefficient _{Tuning}');
ylabel('Number of Neurons');
title('BS-BS cell pairs');

figure(3) % NSBS pairs
NB_histo = histogram(NSBS_pairs, edges)
xlabel('Correlation Coefficient _{Tuning}');
ylabel('Number of Neurons');
title('NS-BS cell pairs');

% plot the histograms also as bar plots for Corel Draw
NB_bars = NB_histo.Values;
BB_bars = BB_histo.Values;

figure(4)
bar(BB_bars, 'FaceColor','k')
xlabel('Correlation Coefficient _{Tuning}');
ylabel('Number of Neurons');
title('BS-BS cell pairs');
set(gca,'XTick',[0.5 1.5 2.5 3.5 4.5]);
set(gca,'XTickLabel',[-1 -0.5 0 0.5 1]);
xlim([0.25 4.75])

figure(5)
bar(NB_bars, 'FaceColor','k')
xlabel('Correlation Coefficient _{Tuning}');
ylabel('Number of Neurons');
title('NS-BS cell pairs');
set(gca,'XTick',[0.5 1.5 2.5 3.5 4.5]);
set(gca,'XTickLabel',[-1 -0.5 0 0.5 1]);
xlim([0.25 4.75])