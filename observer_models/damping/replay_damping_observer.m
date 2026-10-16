function mod_ans_all = replay_damping_observer(outputFile)
% REPLAY_DAMPING_OBSERVER Run the mean-RT60 decision on the exact saved trials.
% Array order: simulation, damping level, model condition, trial replicate.
folder = fileparts(mfilename('fullpath'));
data = load(fullfile(folder,'saved_trials.mat'));
mean1 = mean(data.RT60_1,3);
mean2 = mean(data.RT60_2,3);
correct = double(mean1 > mean2);
ties = mean1 == mean2;
assert(all(data.tie_response(ties)>=0),'Missing saved tie decision.');
correct(ties) = double(data.tie_response(ties));
mod_ans_all = zeros(size(correct,1),5,7,15);
for i = 1:size(correct,1)
    for damping = 1:5
        for condition = 1:7
            indices = find(data.trial_key(i,:,1)==condition & ...
                data.trial_key(i,:,2)==damping);
            assert(numel(indices)==15,'Expected 15 trials per condition/level.');
            mod_ans_all(i,damping,condition,:) = correct(i,indices);
        end
    end
end
root = fileparts(fileparts(folder));
reference = load(fullfile(root,'supplementary_figures','sup figure 8','model_result_200.mat'),'mod_ans_all');
assert(isequal(mod_ans_all,reference.mod_ans_all),'Replay differs from the plotted reference.');
fprintf('Exact replay matches all %d saved model responses.\n',numel(mod_ans_all));
if nargin > 0 && ~isempty(outputFile)
    save(outputFile,'mod_ans_all');
end
end
