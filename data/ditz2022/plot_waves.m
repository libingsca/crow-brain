%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Behavioral Performance %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script plots
% 1) the mean waveform of one narrow and one broad spiking neuron recorded
% from the same electrode.
% 2) normalized average waveforms of a random subset of 300 neurons aligned
% by their minimum.
% 3) the bimodal distribution of waveform durations.

clear, clc
close all

load('data_waves.mat')


%% plot 2B

figure(1)
hold on
plot(subset_300{2,2},'b');
plot(subset_300{2,60},'r');
hold off

ylabel('Normalized amplitude');
xlabel('Time (\mus)');
set(gca,'XTick', [1 7 15 23 31]);
set(gca,'XTickLabel',[-200 0 200 400 600]);

%% plot 2C
for neuron = 1:300
figure(2)
    if grouping(neuron) == 1
        plot(subset_300{2,neuron},'b');
    elseif grouping(neuron) == 2
        plot(subset_300{2,neuron},'r');
    end
    hold on
end
hold off
ylabel('Normalized amplitude');
xlabel('Time (\mus)');
set(gca,'XTick', [1 7 15 23 31]);
set(gca,'XTickLabel',[-200 0 200 400 600]);

%%  plot 2D
% plot histogram with different colors according to the grouping vector
edges = 5:25;
figure(3)
broad_h = histogram(cell2mat(broad_spikes), edges,'FaceColor','b');
set(gca,'XTick',[1 8 16 24]);
set(gca,'XTickLabel',[0 200 400 600]);
hold on
narrow_h = histogram(cell2mat(narrow_spikes), edges,'FaceColor','r');
xlabel('Spike width (\mus)');
ylabel('Number of neurons');
set(gca,'XTick',[1 8 16 24]);
set(gca,'XTickLabel',[0 200 400 600]);
legend('Broad spiking neurons', 'Narrow spiking neurons');
hold off

% same figure as a bar plot instead of histogram
broad_bars = broad_h.Values;
narrow_bars = narrow_h.Values;
figure(4)
bar(broad_bars,'FaceColor','b');
hold on
bar(narrow_bars ,'FaceColor','r');
xlabel('Spike width (\mus)');
ylabel('Number of neurons');
set(gca,'XTick',[3.5 11.5 19.5]);
set(gca,'XTickLabel',[200 400 600]);
legend('Broad spiking neurons', 'Narrow spiking neurons');
hold off