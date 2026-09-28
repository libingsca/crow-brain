%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Physiological differences %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script plots
% 1) mean firing rates during trial phases for NS and BS neurons.
% 2) a cumulative histogram for response latencies.

clear, clc
close all

load('data_physio.mat')

%% plot 3A
% split the spike density vectors in broad and narrow group
broad_group = grouping == 1;
narrow_group = grouping == 2;
narrow_vectors = all_vectors(narrow_group,:);
broad_vectors = all_vectors(broad_group,:);

% take the average spike density in both groups
aver_spk_dens_narrow = sum(narrow_vectors,1)/size(narrow_vectors,1);
aver_spk_dens_broad = sum(broad_vectors,1)/size(broad_vectors,1);

% calculate the SEM for each time point of the average spike density vector
stderror_narrow = zeros(1,2300);
stderror_broad = zeros(1,2300);
for i = 1:2300
    stderror_narrow(i) = std(all_vectors(:,i))/sqrt(size(all_vectors,1));
    stderror_broad(i) = std(all_vectors(:,i))/sqrt(size(all_vectors,1));
end

% smooth the curve in both groups
window = 50; % define window size
smoothed_narrow_spkdens = smoothdata(aver_spk_dens_narrow, 'gaussian',window);
smoothed_broad_spkdens = smoothdata(aver_spk_dens_broad, 'gaussian',window);
smooth_SEM_narrow = smoothdata(stderror_narrow, 'gaussian',window);
smooth_SEM_broad = smoothdata(stderror_broad, 'gaussian',window);

% plot
figure(1)
% SEM area around the curve
x = 1:2300;
curve1 = smoothed_narrow_spkdens-smooth_SEM_narrow;
curve2 = smoothed_narrow_spkdens+smooth_SEM_narrow;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, [1 0.8 0.8]);
hold on
curve1 = smoothed_broad_spkdens-smooth_SEM_broad;
curve2 = smoothed_broad_spkdens+smooth_SEM_broad;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, [0.8 0.8 1]);

% average firing rate of NS and BS neurons
plot(smoothed_narrow_spkdens, 'color', 'r', 'LineWidth', 2)
xlabel('Time [ms]')
ylabel('Mean firing rate [Hz]')
xlim([0 2300])
xticks([0 500 1300 2300])
xticklabels({'-500','0','800','1800'})
plot(smoothed_broad_spkdens, 'color', 'b', 'LineWidth', 2);
xline(500);
xline(1300);
legend('Narrow spiking Neurons', 'Broad spiking neurons')
hold off

%% plot 3B
% preallocate space
NS_latencies = zeros(1,218);
BS_latencies = zeros(1,918);
a = 1;
b = 1;

% loop through all neurons
for neuron = 1:1136
    
    % change these varibales according to the data set (1-3)
    if neuron <= 824 % data set 1 and 2
        baseline = 400:600;
        analysisperiod = 600:1400-5;
        sampleonset = 600;
    elseif neuron >= 825 % data set 3
        baseline = 300:500;
        analysisperiod = 500:1300-5;
        sampleonset = 500;
    end %if
    
    % get the average spike density vector (unsmoothed) for each neuron
    spk_density = spk_densities{neuron};
    
    % smooth this vector with a sliding window and a gaussian kernel of 10 ms
    spk_density = smoothdata(spk_density, 'gaussian', 10);
    
    % determine maximum of baseline period (200ms before stimulus onset)
    threshold = max(spk_density(baseline));
    
    % find the first 5 consecutive time bins after stimulus onset,
    % that pass the threshold
    % in case threshold is not passed, set a default value of NaN
    latency = NaN;
    for bin = analysisperiod
        bin_count = 0;
        if spk_density(bin) > threshold
            bin_count = 1; % count if all 5 bins are higher
            for i = 1:4
                if spk_density(bin+i) > threshold
                    bin_count = bin_count + 1;
                end
            end
            if bin_count >= 5
                % the first of these bin is is defined as the latency
                % bin minus sample onset, to get the actual latency
                latency = bin-sampleonset;
                break
            end % if
        end % if 
    end % for bins
    
    % decide if this neurons was NS or BS and save the latency value in the
    % according vector
    if grouping(neuron) == 1 %BS
        BS_latencies(a) = latency;
        a = a+1;
    elseif grouping(neuron) == 2 % NS
        NS_latencies(b) = latency;
        b = b+1;
    end
    
end % end for neuron

% disregard all cells with latencies < 50 and > 250
NS_disregard = find(NS_latencies < 50 | NS_latencies > 250);
NS_latencies(NS_disregard) = nan;

BS_disregard = find(BS_latencies < 50 | BS_latencies > 250);
BS_latencies(BS_disregard) = nan;

% plot the cumulative histogram for NS and BS neurons
figure(1)
plotNS = cdfplot(NS_latencies)
hold on
plotBS = cdfplot(BS_latencies)
legend('Narrow spiking neurons','Broad spiking neurons')
xlabel('Visual response latencies (ms)')
ylabel('Proportion of cells')
set(plotNS, 'Color', 'r', 'LineWidth', 2)
set(plotBS, 'Color', 'b', 'LineWidth', 2)