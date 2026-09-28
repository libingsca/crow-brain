%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Firing rate modulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script plots
% 1) responses of an example BS neuron and its average action potential
% waveform.
% 2) responses of an example NS neuron and its average action potential
% waveform.
% 3) average spike historgrams of BS and NS neurons that were selective for
% numerosities 1-30 during sample and delay period.

clear, clc
close all

load('data_firingrate.mat')

%% plot 4A and 4B
% dot raster and spike desnity of one NS-BS example pair, recorded at same
% electrode, numerosity tuned
offset = 0; % we introduce an offset of 300, because we want to use a 
% sliding window smoothing function and need to start before the actual
% start of the trial

% colour of dotraster plot
% enter as many colours as conditions of experiment
cmap =  lines(7);

% create plot for 1st neuron: 290
figure(1)
subplot(2,2,1)
dotraster_num7(recording1, corr_trials, offset, cmap)
subplot(2,2,3)
spikedensity_num7(recording1, corr_trials, offset, cmap)
% plot waveforms of neuron 291 as an insert into density histogram
axes('Position', [0.2 0.3 0.1 0.1])
box on
plot(normed_waves{468+290},'r'); % 468 neurons in first data set
ylabel('Amplitude');
xlabel('Time (\mus)');
set(gca,'XTick', [8 24]);
set(gca,'XTickLabel',[0 400]);
set(gca,'YTick', [-1000 0 1000]);
set(gca,'YTickLabel',[-0.4 0 0.4]);

% create plot for 2nd neuron: neuron 291
subplot(2,2,2)
dotraster_num7(recording2, corr_trials2, offset, cmap)
subplot(2,2,4)
spikedensity_num7(recording2, corr_trials2, offset, cmap)

% plot waveforms of neuron 291 as an insert into density histogram
axes('Position', [0.8 0.3 0.1 0.1])
box on
plot(normed_waves{468+291},'b'); % 468 neurons in first data set
ylabel('Amplitude');
xlabel('Time (\mus)');
set(gca,'XTick', [8 24]);
set(gca,'XTickLabel',[0 400]);
set(gca,'YTick', [-1000 0 1000]);
set(gca,'YTickLabel',[-0.4 0 0.4]);

%% plot 4C and 4D

% pick the colors for the plot
b1=[0 0 1]; b2=[0.3216 0.3216 1]; b3=[0.4588 0.4588 1]; b4=[0.6314 0.6314 1]; b5=[0.7 0.7 1]; b6=[0.8 0.8 1]; b7=[0.95 0.95 1];
r1=[1 0 0]; r2=[1 0.3216 0.3216]; r3=[1 0.4588 0.4588]; r4=[1 0.6314 0.6314]; r5=[1 0.7 0.7]; r6=[1 0.8 0.8]; r7=[1 0.95 0.95];

for phase = 1:2 % sample and delay phase
    
    % loop through all numerosities: first loop = most preferred, second
    % loop = second most preferred etc.
    for numeros = 1:7
        
        m = 1; % counting variable for spike density vectors sorted into preferences (sample and broad)
        n = 1; % counting variable for spike density vectors sorted into preferences (delay and broad)
        o = 1; % counting variable for spike density vectors sorted into preferences (sample and narrow)
        p = 1; % counting variable for spike density vectors sorted into preferences (delay and narrow)
        
        for neuron = 487:824 % 338 neurons
            % Get the spike density vector of each numerosity and each neuron and
            % save it into 7 matrices. The first matrix contains all vectors with
            % the most preferred numerosity, the second matrix contains all spike
            % density vectors of the second preferred numerosity etc.
            
            if phase == 1 && numerosity{neuron}{1} == 1 && grouping(neuron) == 1 % if sample, numerosity, broad
                pref = preference{neuron}(1,numeros); % get the preference of this neuron
                broad_sample_preference{numeros}(m,:) = firing_numer{neuron}(pref,:);
                m = m + 1;
            end % if
            
            if phase == 1 && numerosity{neuron}{1} == 1 && grouping(neuron) == 2 % if sample, numerosity, narrow
                pref = preference{neuron}(1,numeros); % get the preference of this neuron
                narrow_sample_preference{numeros}(o,:) = firing_numer{neuron}(pref,:);
                o = o + 1;
            end % if
            
            if phase == 2 && numerosity{neuron}{2} == 1 && grouping(neuron) == 1 % if delay, numerosity, broad
                pref = preference{neuron}(1,numeros); % get the preference of this neuron
                broad_delay_preference{numeros}(n,:) = firing_numer{neuron}(pref,:);
                n = n + 1;
            end % if
            
            if phase == 2 && numerosity{neuron}{2} == 1 && grouping(neuron) == 2 % if sample, numerosity, narrow
                pref = preference{neuron}(1,numeros); % get the preference of this neuron
                narrow_delay_preference{numeros}(p,:) = firing_numer{neuron}(pref,:);
                p = p + 1;
            end % if
            
        end % for neuron
    end % for numeros
end % for phase

% now calculate the mean spike density vector for the preferred
% numerosities, split in sample, delay and broad and narrow
broad_sample_most_pref = sum(broad_sample_preference{1},1)/size(broad_sample_preference{1},1);
broad_sample_scnd_most_pref = sum(broad_sample_preference{2},1)/size(broad_sample_preference{2},1);
broad_sample_thrd_most_pref = sum(broad_sample_preference{3},1)/size(broad_sample_preference{3},1);
broad_sample_frth_most_pref = sum(broad_sample_preference{4},1)/size(broad_sample_preference{4},1);
broad_sample_fivth_most_pref = sum(broad_sample_preference{5},1)/size(broad_sample_preference{5},1);
broad_sample_sixth_most_pref = sum(broad_sample_preference{6},1)/size(broad_sample_preference{6},1);
broad_sample_least_pref = sum(broad_sample_preference{7},1)/size(broad_sample_preference{7},1);

narrow_sample_most_pref = sum(narrow_sample_preference{1},1)/size(narrow_sample_preference{1},1);
narrow_sample_scnd_most_pref = sum(narrow_sample_preference{2},1)/size(narrow_sample_preference{2},1);
narrow_sample_thrd_most_pref = sum(narrow_sample_preference{3},1)/size(narrow_sample_preference{3},1);
narrow_sample_frth_most_pref = sum(narrow_sample_preference{4},1)/size(narrow_sample_preference{4},1);
narrow_sample_fivth_most_pref = sum(narrow_sample_preference{5},1)/size(narrow_sample_preference{5},1);
narrow_sample_sixth_most_pref = sum(narrow_sample_preference{6},1)/size(narrow_sample_preference{6},1);
narrow_sample_least_pref = sum(narrow_sample_preference{7},1)/size(narrow_sample_preference{7},1);

broad_delay_most_pref = sum(broad_delay_preference{1},1)/size(broad_delay_preference{1},1);
broad_delay_scnd_most_pref = sum(broad_delay_preference{2},1)/size(broad_delay_preference{2},1);
broad_delay_thrd_most_pref = sum(broad_delay_preference{3},1)/size(broad_delay_preference{3},1);
broad_delay_frth_most_pref = sum(broad_delay_preference{4},1)/size(broad_delay_preference{4},1);
broad_delay_fivth_most_pref = sum(broad_delay_preference{5},1)/size(broad_delay_preference{5},1);
broad_delay_sixth_most_pref = sum(broad_delay_preference{6},1)/size(broad_delay_preference{6},1);
broad_delay_least_pref = sum(broad_delay_preference{7},1)/size(broad_delay_preference{7},1);

narrow_delay_most_pref = sum(narrow_delay_preference{1},1)/size(narrow_delay_preference{1},1);
narrow_delay_scnd_most_pref = sum(narrow_delay_preference{2},1)/size(narrow_delay_preference{2},1);
narrow_delay_thrd_most_pref = sum(narrow_delay_preference{3},1)/size(narrow_delay_preference{3},1);
narrow_delay_frth_most_pref = sum(narrow_delay_preference{4},1)/size(narrow_delay_preference{4},1);
narrow_delay_fivth_most_pref = sum(narrow_delay_preference{5},1)/size(narrow_delay_preference{5},1);
narrow_delay_sixth_most_pref = sum(narrow_delay_preference{6},1)/size(narrow_delay_preference{6},1);
narrow_delay_least_pref = sum(narrow_delay_preference{7},1)/size(narrow_delay_preference{7},1);

% smooth the curves
window = 500; % define window size
smoothed_broad_sample_most_pref = smoothdata(broad_sample_most_pref, 'gaussian',window);
smoothed_broad_sample_scnd_most_pref = smoothdata(broad_sample_scnd_most_pref, 'gaussian',window);
smoothed_broad_sample_thrd_most_pref = smoothdata(broad_sample_thrd_most_pref, 'gaussian',window);
smoothed_broad_sample_frth_most_pref = smoothdata(broad_sample_frth_most_pref, 'gaussian',window);
smoothed_broad_sample_fivth_most_pref = smoothdata(broad_sample_fivth_most_pref, 'gaussian',window);
smoothed_broad_sample_sixth_most_pref = smoothdata(broad_sample_sixth_most_pref, 'gaussian',window);
smoothed_broad_sample_least_pref = smoothdata(broad_sample_least_pref, 'gaussian',window);

smoothed_narrow_sample_most_pref = smoothdata(narrow_sample_most_pref, 'gaussian',window);
smoothed_narrow_sample_scnd_most_pref = smoothdata(narrow_sample_scnd_most_pref, 'gaussian',window);
smoothed_narrow_sample_thrd_most_pref = smoothdata(narrow_sample_thrd_most_pref, 'gaussian',window);
smoothed_narrow_sample_frth_most_pref = smoothdata(narrow_sample_frth_most_pref, 'gaussian',window);
smoothed_narrow_sample_fivth_most_pref = smoothdata(narrow_sample_fivth_most_pref, 'gaussian',window);
smoothed_narrow_sample_sixth_most_pref = smoothdata(narrow_sample_sixth_most_pref, 'gaussian',window);
smoothed_narrow_sample_least_pref = smoothdata(narrow_sample_least_pref, 'gaussian',window);

smoothed_broad_delay_most_pref = smoothdata(broad_delay_most_pref, 'gaussian',window);
smoothed_broad_delay_scnd_most_pref = smoothdata(broad_delay_scnd_most_pref, 'gaussian',window);
smoothed_broad_delay_thrd_most_pref = smoothdata(broad_delay_thrd_most_pref, 'gaussian',window);
smoothed_broad_delay_frth_most_pref = smoothdata(broad_delay_frth_most_pref, 'gaussian',window);
smoothed_broad_delay_fivth_most_pref = smoothdata(broad_delay_fivth_most_pref, 'gaussian',window);
smoothed_broad_delay_sixth_most_pref = smoothdata(broad_delay_sixth_most_pref, 'gaussian',window);
smoothed_broad_delay_least_pref = smoothdata(broad_delay_least_pref, 'gaussian',window);

smoothed_narrow_delay_most_pref = smoothdata(narrow_delay_most_pref, 'gaussian',window);
smoothed_narrow_delay_scnd_most_pref = smoothdata(narrow_delay_scnd_most_pref, 'gaussian',window);
smoothed_narrow_delay_thrd_most_pref = smoothdata(narrow_delay_thrd_most_pref, 'gaussian',window);
smoothed_narrow_delay_frth_most_pref = smoothdata(narrow_delay_frth_most_pref, 'gaussian',window);
smoothed_narrow_delay_fivth_most_pref = smoothdata(narrow_delay_fivth_most_pref, 'gaussian',window);
smoothed_narrow_delay_sixth_most_pref = smoothdata(narrow_delay_sixth_most_pref, 'gaussian',window);
smoothed_narrow_delay_least_pref = smoothdata(narrow_delay_least_pref, 'gaussian',window);

% plot the spike density vectors

% sample, numerosity 1-30
figure(2)
hold on
plot(smoothed_broad_sample_most_pref, 'color', b1, 'LineWidth', 2.0)
plot(smoothed_broad_sample_scnd_most_pref, 'color', b2, 'LineWidth', 2.0)
plot(smoothed_broad_sample_thrd_most_pref, 'color', b3, 'LineWidth', 2.0)
plot(smoothed_broad_sample_frth_most_pref, 'color', b4, 'LineWidth', 2.0)
plot(smoothed_broad_sample_fivth_most_pref, 'color', b5, 'LineWidth', 2.0)
plot(smoothed_broad_sample_sixth_most_pref, 'color', b6, 'LineWidth', 2.0)
plot(smoothed_broad_sample_least_pref, 'color', b7, 'LineWidth', 2.0)
plot(smoothed_narrow_sample_most_pref, 'color', r1, 'LineWidth', 2.0)
plot(smoothed_narrow_sample_scnd_most_pref, 'color', r2, 'LineWidth', 2.0)
plot(smoothed_narrow_sample_thrd_most_pref, 'color', r3, 'LineWidth', 2.0)
plot(smoothed_narrow_sample_frth_most_pref, 'color', r4, 'LineWidth', 2.0)
plot(smoothed_narrow_sample_fivth_most_pref, 'color', r5, 'LineWidth', 2.0)
plot(smoothed_narrow_sample_sixth_most_pref, 'color', r6, 'LineWidth', 2.0)
plot(smoothed_narrow_sample_least_pref, 'color', r7, 'LineWidth', 2.0)
xline(600);
xline(1400);
hold off
xlabel('Time [ms]')
ylabel('Mean firing rate [Hz]')
xlim([0 2400])
xticks([0 600 1400 2400])
xticklabels({'-600','0','800','1800'})
ylim([1 11])
title('Sample-selective neurons')

% delay, numerosity 1-30
figure(3)
hold on
plot(smoothed_broad_delay_most_pref, 'color', b1, 'LineWidth', 2.0)
plot(smoothed_broad_delay_scnd_most_pref, 'color', b2, 'LineWidth', 2.0)
plot(smoothed_broad_delay_thrd_most_pref, 'color', b3, 'LineWidth', 2.0)
plot(smoothed_broad_delay_frth_most_pref, 'color', b4, 'LineWidth', 2.0)
plot(smoothed_broad_delay_fivth_most_pref, 'color', b5, 'LineWidth', 2.0)
plot(smoothed_broad_delay_sixth_most_pref, 'color', b6, 'LineWidth', 2.0)
plot(smoothed_broad_delay_least_pref, 'color', b7, 'LineWidth', 2.0)
plot(smoothed_narrow_delay_most_pref, 'color', r1, 'LineWidth', 2.0)
plot(smoothed_narrow_delay_scnd_most_pref, 'color', r2, 'LineWidth', 2.0)
plot(smoothed_narrow_delay_thrd_most_pref, 'color', r3, 'LineWidth', 2.0)
plot(smoothed_narrow_delay_frth_most_pref, 'color', r4, 'LineWidth', 2.0)
plot(smoothed_narrow_delay_fivth_most_pref, 'color', r5, 'LineWidth', 2.0)
plot(smoothed_narrow_delay_sixth_most_pref, 'color', r6, 'LineWidth', 2.0)
plot(smoothed_narrow_delay_least_pref, 'color', r7, 'LineWidth', 2.0)
xline(600);
xline(1400);
hold off
xlabel('Time [ms]')
ylabel('Mean firing rate [Hz]')
xlim([0 2400])
xticks([0 600 1400 2400])
xticklabels({'-600','0','800','1800'})
ylim([1 11])
title('Delay-selective neurons')
