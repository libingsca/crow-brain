%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Behavioral Performance %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script plots the average performance curves for a given sample
% numerosity in experiments with 4, 5 and 30 numerosities.

clear, clc
close all

load('data_behavior.mat')

%% plot 1C
figure(1)
hold on
plot(all_numer_1_3, 'color', [0.9350 0.0780 0.1840], 'LineWidth', 2)
plot(all_numer_2_3, 'color', [1.0000 0.4118 0.1608], 'LineWidth', 2)
plot(all_numer_3_3, 'color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(all_numer_4_3, 'color', [0.3010 0.7450 0.9330], 'LineWidth', 2)
plot(all_average_3, 'color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 2)
hold off

% legend and axes
xlabel('Numerosity')
ylabel('Judged same as sample (%)')
xlim([0.9 4.1])
set(gca,'XTick',[1 2 3 4]);
legend('Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Average','Location', 'northeastoutside')

%% plot 1D
figure(2)
hold on
plot(all_numer_1_1, 'color', [0.9350 0.0780 0.1840], 'LineWidth', 2)
plot(all_numer_2_1, 'color', [1.0000 0.4118 0.1608], 'LineWidth', 2)
plot(all_numer_3_1, 'color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(all_numer_4_1, 'color', [0.3010 0.7450 0.9330], 'LineWidth', 2)
plot(all_numer_5_1, 'color', [0.0902 0.3333 1.0000], 'LineWidth', 2)
plot(all_average_1, 'color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 2)
hold off

% legend and axes
xlabel('Numerosity')
ylabel('Judged same as sample (%)')
xlim([0.9 5.1])
set(gca,'XTick',[1 2 3 4 5]);
legend('Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Average','Location', 'northeastoutside')

%% plot 1E
figure(3)
hold on
plot(all_numer_1_2, 'color', [0.9350 0.0780 0.1840], 'LineWidth', 2)
plot(all_numer_2_2, 'color', [1.0000 0.4118 0.1608], 'LineWidth', 2)
plot(all_numer_4_2, 'color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
plot(all_numer_7_2, 'color', [0.3010 0.7450 0.9330], 'LineWidth', 2)
plot(all_numer_12_2, 'color', [0.0902 0.3333 1.0000], 'LineWidth', 2)
plot(all_numer_20_2, 'color', [0 0.0353 0.5098], 'LineWidth', 2)
plot(all_numer_30_2, 'color', [0 0.4902 0.2627], 'LineWidth', 2)
plot(all_average_2, 'color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 2)
hold off

% legend and axes
xlabel('Numerosity')
ylabel('Judged same as sample (%)')
xlim([0.9 7.1])
set(gca,'XTick',[1 2 3 4 5 6 7]);
set(gca,'XTickLabel',[1 2 4 7 12 20 30]);
legend('Sample 1', 'Sample 2', 'Sample 4', 'Sample 7', 'Sample 12', 'Sample 20', 'Sample 30', 'Average','Location', 'northeastoutside')
