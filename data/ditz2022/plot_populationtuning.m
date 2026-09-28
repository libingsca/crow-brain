%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Population tuning curves %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script plots population tuning curves that were numerosity selective
% during the sample/delay period

clear, clc
close all

load('data_tuningcurve.mat')

%% plot 5A until 5D

% normalize the data
for i = 1:7
norm_NS_sample_small(:,i) = (NS_sample_tuning_small(:,i)./NS_sample_tuning_small(:,4))*100;
norm_BS_sample_small(:,i) = (BS_sample_tuning_small(:,i)./BS_sample_tuning_small(:,4))*100;
norm_NS_delay_small(:,i) = (NS_delay_tuning_small(:,i)./NS_delay_tuning_small(:,4))*100;
norm_BS_delay_small(:,i) = (BS_delay_tuning_small(:,i)./BS_delay_tuning_small(:,4))*100;
end

for i = 1:9
norm_NS_sample_large(:,i) = (NS_sample_tuning_large(:,i)./NS_sample_tuning_large(:,5))*100;
norm_BS_sample_large(:,i) = (BS_sample_tuning_large(:,i)./BS_sample_tuning_large(:,5))*100;
norm_NS_delay_large(:,i) = (NS_delay_tuning_large(:,i)./NS_delay_tuning_large(:,5))*100;
norm_BS_delay_large(:,i) = (BS_delay_tuning_large(:,i)./BS_delay_tuning_large(:,5))*100;
end

% calculate the mean firing rate for each numerosity in the tuning curve
% and each cell group
mean_NS_sample_tuning_small = mean(norm_NS_sample_small, 1, 'omitnan');
mean_BS_sample_tuning_small = mean(norm_BS_sample_small, 1, 'omitnan');
mean_NS_delay_tuning_small = mean(norm_NS_delay_small, 1, 'omitnan');
mean_BS_delay_tuning_small = mean(norm_BS_delay_small, 1, 'omitnan');

mean_NS_sample_tuning_large = mean(norm_NS_sample_large, 1, 'omitnan');
mean_BS_sample_tuning_large = mean(norm_BS_sample_large, 1, 'omitnan');
mean_NS_delay_tuning_large = mean(norm_NS_delay_large, 1, 'omitnan');
mean_BS_delay_tuning_large = mean(norm_BS_delay_large, 1, 'omitnan');

% calculate error bars
sem_NS_sample_tuning_small = std(norm_NS_sample_small,1, 'omitnan')/ sqrt(size(norm_NS_sample_small,1));
sem_BS_sample_tuning_small = std(norm_BS_sample_small,1, 'omitnan')/ sqrt(size(norm_BS_sample_small,1));
sem_NS_delay_tuning_small = std(norm_NS_delay_small,1,'omitnan')/ sqrt(size(norm_NS_delay_small,1));
sem_BS_delay_tuning_small = std(norm_BS_delay_small,1,'omitnan')/ sqrt(size(norm_BS_delay_small,1));

sem_NS_sample_tuning_large = std(norm_NS_sample_large,1, 'omitnan')/ sqrt(size(norm_NS_sample_large,1));
sem_BS_sample_tuning_large = std(norm_BS_sample_large,1, 'omitnan')/ sqrt(size(norm_BS_sample_large,1));
sem_NS_delay_tuning_large = std(norm_NS_delay_large,1,'omitnan')/ sqrt(size(norm_NS_delay_large,1));
sem_BS_delay_tuning_large = std(norm_BS_delay_large,1,'omitnan')/ sqrt(size(norm_BS_delay_large,1));

% small numerosities
figure(1)
errorbar(mean_NS_sample_tuning_small,sem_NS_sample_tuning_small, '-r.')
hold on
errorbar(mean_BS_sample_tuning_small, sem_BS_sample_tuning_small, '-b.')
hold off
title('Small numerosities, sample')
xlabel('Numerical distance')
ylabel('Normalized response')
ylim([0 100])
xlim([0.9 7.1])
set(gca,'XTick',1:7);
set(gca,'XTickLabel',[-3 -2 -1 0 1 2 3]);


figure(2)
errorbar(mean_NS_delay_tuning_small,sem_NS_delay_tuning_small, '-r.')
hold on
errorbar(mean_BS_delay_tuning_small,sem_BS_delay_tuning_small, '-b.')
hold off
title('Small numerosities, delay')
xlabel('Numerical distance')
ylabel('Normalized response')
ylim([0 100])
xlim([0.9 7.1])
set(gca,'XTick',1:7);
set(gca,'XTickLabel',[-3 -2 -1 0 1 2 3]);

% large numerosities
figure(3)
errorbar(mean_NS_sample_tuning_large(1:9),sem_NS_sample_tuning_large(1:9), '-r.')
hold on
errorbar(mean_BS_sample_tuning_large(1:9),sem_BS_sample_tuning_large(1:9), '-b.')
hold off
title('Large numerosities, sample')
xlabel('Normalized numerical distance')
ylabel('Normalized response')
ylim([0 100])
xlim([0.9 9.1])
set(gca,'XTick',1:9);
set(gca,'XTickLabel',[-4 -3 -2 -1 0 1 2 3 4]);

figure(4)
errorbar(mean_NS_delay_tuning_large(1:9),sem_NS_delay_tuning_large(1:9), '-r.')
hold on
errorbar(mean_BS_delay_tuning_large(1:9),sem_BS_delay_tuning_large(1:9), '-b.')
hold off
title('Large numerosities, delay')
xlabel('Normalized numerical distance')
ylabel('Normalized response')
ylim([0 100])
xlim([0.9 9.1])
set(gca,'XTick',1:9);
set(gca,'XTickLabel',[-4 -3 -2 -1 0 1 2 3 4]);
