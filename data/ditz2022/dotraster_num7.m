function dotraster_num7(recording, corr_trials, offset, varargin)
% dotraster creates dot raster plots, one for each condition
%
% INPUT:
% recording:   spike table of recorded neuron
% corr_trials: information about all the correct trials
% offset:      when does the recording start, in relation to stimulus onset
% varargin:    if colormap is entered, then use it (what color should the
%              plot have)
%
% OUTPUT:      dotraster plot
%% preallocate space and define variables
gap = 0;

% get the groups (numerosities) of the experiment: 1-7
% get ony correct trials and for eah trial the tested numerosity
steppp = 0;
grps = nan(size(corr_trials,1),7);
for i = 1:7
grps(:,i) = (corr_trials(:,4) == 1 & 10*corr_trials(:,2)+corr_trials(:,3) >= 01+steppp & 10*corr_trials(:,2)+corr_trials(:,3) <= 07+steppp) ...
    | (corr_trials(:,4) == 1 & 10*corr_trials(:,2)+corr_trials(:,3) >= 50+steppp & 10*corr_trials(:,2)+corr_trials(:,3) <= 56+steppp);
steppp = steppp + 7;
end
         
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
%% display dotraster for different conditions

% cut recording matrix and matrix with only correct trials into correct size
% [cutter, ~] = find(recording);
% cutter = sort(cutter);
% recording(1:cutter(1)-1,:) = [];
% recording(cutter(end)+1:end,:) = [];
for i = 1:7
% get only correct trials and trials in single positions
% then look at part of matrix for further computation:
% save at what time spike occurs and save at what trial spike occurs
temp_recording = recording(find(grps(:,i)),:);

[trialspike, timespike] = find(temp_recording);

% display dotraster for each condition
plot(timespike, trialspike+gap, '.', 'Color', cmap(i,:));

hold on

% save trialspike values of previous loop iterations and add them,
% add 5 for a gap between each condition
gap = gap + max(trialspike) + 5;
tick(i) = gap; 
end

% define figure chacateristics
ylim([0 max(gap)-5]);
set(gca,'YTick',[tick-50]);
set(gca,'YTickLabel',[1 2 4 7 12 20 30])
set(gca,'XTick',[0 600 1400 2400]);% 0-600 is presample, 600+800=1400 is
% sample, 1400+1000=2400 is delay
set(gca,'XTickLabel',[-600 0 800 1800])
xlabel('Time [ms]')
xlim([150 2400]);
hold off
end
