function result = compute_exp0_statistics(nResamples)
% COMPUTE_EXP0_STATISTICS MATLAB counterpart of the existing Experiment 0 Python analysis.
% The master statistics workflow now runs entirely in MATLAB.
if nargin < 1, nResamples = 10000; end
validateattributes(nResamples, {'numeric'}, {'scalar','integer','positive'});
folder = fileparts(mfilename('fullpath'));
sourceFile = fullfile(folder,'Experiment_0_data.csv');
data = readtable(sourceFile,'VariableNamingRule','preserve');
materials = ["metal","glass","ceramic","stone","wood","plastic","cardboard"];
variables = string(data.Properties.VariableNames);
trialColumns = variables(~cellfun('isempty',regexp(cellstr(variables),'^trial_\d+_condition$','once')));
n = height(data); counts = zeros(n,3,49);
for column = trialColumns
    prefix = extractBefore(column,'_condition');
    [validTruth,truth] = ismember(lower(string(data.(prefix+"_true"))),materials);
    [validResponse,response] = ismember(lower(string(data.(prefix+"_response"))),materials);
    condition = double(data.(column))+1;
    assert(all(validTruth & validResponse & ismember(condition,1:3)), 'Invalid Experiment 0 trial.');
    for participant = 1:n
        cellIndex = sub2ind([7 7],truth(participant),response(participant));
        counts(participant,condition(participant),cellIndex) = ...
            counts(participant,condition(participant),cellIndex)+1;
    end
end
assert(all(sum(counts,3)==56,'all'), 'Expected 56 trials per participant and condition.');
total = reshape(sum(counts,1),3,49);
correlations = row_correlations(total(1,:),total);
observed = correlations(2)-correlations(3);
stream = RandStream('mt19937ar','Seed',260900);
nullDifferences = zeros(nResamples,1); bootstrapDifferences = zeros(nResamples,1);
for b = 1:nResamples
    indices = randi(stream,n,[n 1]);
    bootTotal = reshape(sum(counts(indices,:,:),1),3,49);
    bootCorr = row_correlations(bootTotal(1,:),bootTotal);
    bootstrapDifferences(b) = bootCorr(2)-bootCorr(3);
    pair = counts(:,2:3,:);
    swap = rand(stream,n,1)<0.5;
    pair(swap,:,:) = pair(swap,[2 1],:);
    pc = row_correlations(total(1,:),reshape(sum(pair,1),2,49));
    nullDifferences(b) = pc(1)-pc(2);
end
assert(all(isfinite(bootstrapDifferences)) && all(isfinite(nullDifferences)));
ci = prctile(bootstrapDifferences,[2.5 97.5]);
p = (1+sum(abs(nullDifferences)>=abs(observed)-1e-12))/(nResamples+1);
Experiment = "0"; Condition = "Resynthesized IRs"; Reference = "Full generative model";
N = n; EffectMetric = "r(Recorded,Resynthesized) - r(Recorded,Full)";
EffectEstimate = observed; CILower = ci(1); CIUpper = ci(2);
PValue = p; TestStatistic = observed; DF = NaN; Status = "ok; retained legacy Experiment 0 comparison";
result.table = table(Experiment,Condition,Reference,N,EffectMetric,EffectEstimate, ...
    CILower,CIUpper,PValue,TestStatistic,DF,Status);
result.conditionCorrelations = table(["Recorded";"Resynthesized IRs";"Full generative model"], ...
    correlations','VariableNames',{'Condition','CorrelationWithRecorded'});
result.confusionCounts = counts;
result.bootstrapDifferences = bootstrapDifferences;
result.nullDifferences = nullDifferences;
result.sourceFile = sourceFile; result.seed = 260900; result.nResamples = nResamples;
result.method = "Legacy Experiment 0 comparison ported to MATLAB; two-sided whole-condition participant permutation, (b+1)/(B+1); added paired participant-bootstrap 95% percentile CI.";
result.demographics = struct('NMetadata',height(data),'MeanAge',mean(data.age,'omitnan'), ...
    'SDAge',std(data.age,'omitnan'));
result.table.AnalysisSet = repmat("Additional comparison",height(result.table),1);
result.table.TestType = repmat("Participant permutation test",height(result.table),1);
result.table.StatisticName = repmat("Correlation difference",height(result.table),1);
result.table.AnalysisSet(:) = "Existing comparison; added CI";
result.existingTests = result.table;
% Preserve the gender summary printed by the original Experiment 0 script.
[gender,~,group] = unique(string(data.gender));
result.genderCounts = table(gender,accumarray(group,1),'VariableNames',{'Gender','Count'});
disp(result.table);
end

function r = row_correlations(reference,rows)
reference = reference-mean(reference); rows = rows-mean(rows,2);
r = (rows*reference')'./sqrt(sum(rows.^2,2)'*sum(reference.^2));
end
