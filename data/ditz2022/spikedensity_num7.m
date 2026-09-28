function spikedensity_num7(recording, corr_trials, offset, varargin)
% This function creates smoothed spike density plots.
%
% INPUT:
% recording:   spike table of recorded neuron
% corr_trials: information about all the correct trials
% offset:      when does the recording start, in relation to stimulus onset
% varargin:    if colormap is entered, then use it (what color shoulod the
%              plot have)
%
% OUTPUT:      smoothed spike density plot


%% define variables
windowsize = 300;
stepsize = 1;

% set color for plot
switch nargin
    case 4
        cmap = varargin{1};
    case 3
        cmap = [0    0.4470    0.7410;
            0.8500    0.3250    0.0980;
            0.9290    0.6940    0.1250;
            0.4940    0.1840    0.5560;
            0.4660    0.6740    0.1880;
            0.3010    0.7450    0.9330;
            0.6350    0.0780    0.1840;
            1         0         0.4];
end
%% create the spike density vector
steppp = 0;
grps = nan(size(corr_trials,1),7);
for i = 1:7  
grps(:,i) = (corr_trials(:,4) == 1 & 10*corr_trials(:,2)+corr_trials(:,3) >= 01+steppp & 10*corr_trials(:,2)+corr_trials(:,3) <= 07+steppp) ...
    | (corr_trials(:,4) == 1 & 10*corr_trials(:,2)+corr_trials(:,3) >= 50+steppp & 10*corr_trials(:,2)+corr_trials(:,3) <= 56+steppp);
steppp = steppp + 7;

    temp_recording = recording(find(grps(:,i)),:);
    vector = sum(temp_recording)/size(temp_recording,1)*1000;
    
    %% smooth the vector
    % smoothed by a sliding average
    smoothed = smoothing(vector, windowsize, stepsize);
    
    %% plot the smoothed vector
    plot(smoothed, 'Color', cmap(i,:))
    xlabel('Time [ms]')
    ylabel('Firing rate [Hz]')
    set(gca,'XTick',[0 600 1400 2400]);% 0-600 is presample, 600+800=1400 is
    % sample, 1400+1000=2400 is delay
    set(gca,'XTickLabel',[-600 0 800 1800])
    xlim([150 2400]);
    hold on
end
hold off
end