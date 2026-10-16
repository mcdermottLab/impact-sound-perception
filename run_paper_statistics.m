function results = run_paper_statistics(nResamples,outputDir,exp1Reference)
% RUN_PAPER_STATISTICS Run all statistics and export one consolidated CSV plus MAT results.
%   run_paper_statistics                         % 10,000 draws, default output folder
%   results = run_paper_statistics(10000,outputDir,"chance")
% Experiment 1 compares each condition against chance (0.5).
% Original comparisons run alongside additions in supplementary_statistics.csv.
% Detailed tables and resampling distributions remain in the MAT results.
if nargin < 1, nResamples = 10000; end
validateattributes(nResamples, {'numeric'}, {'scalar','integer','positive'});
root = fileparts(mfilename('fullpath'));
if nargin < 2 || isempty(outputDir) || strlength(string(outputDir))==0
    outputDir = fullfile(root,'stats_results');
end
if nargin < 3, exp1Reference = "chance"; end
if ~isfolder(outputDir), mkdir(outputDir); end
originalPath = path;
cleanupPath = onCleanup(@() path(originalPath));
experiments = ["0","1","2","3a","3b","3c","4"];
for experiment = experiments
    addpath(fullfile(root,'Human_Experiments',"Experiment "+experiment));
end
results = struct;
results.runInfo = struct('MATLABVersion',version,'CreatedAt',char(datetime('now')), ...
    'Resamples',nResamples,'ConfidenceLevel',0.95,'PValues','Unadjusted, two-sided', ...
    'Exp1Reference',string(exp1Reference));
combined = table;
existingCombined = table;
demographics = table;
for experiment = experiments
    fprintf('\nExperiment %s: %d resamples\n',experiment,nResamples);
    if experiment=="1"
        result = compute_exp1_statistics(nResamples,exp1Reference);
    else
        result = feval("compute_exp"+experiment+"_statistics",nResamples);
    end
    results.("Exp"+experiment) = result;
    combined = [combined; final_comparisons(result)]; %#ok<AGROW>
    if isfield(result,'existingTests')
        existingCombined = [existingCombined; result.existingTests]; %#ok<AGROW>
    end
    demographicBreakdown = demographic_summary(root,experiment);
    demographics = [demographics; demographicBreakdown]; %#ok<AGROW>
    result.demographicBreakdown = demographicBreakdown;
    fprintf('Demographics:\n');
    for k = 1:height(demographicBreakdown)
        fprintf('  %s: N=%d, M=%d, F=%d, Other=%d, Age mean=%.2f, SD=%.2f\n', ...
            demographicBreakdown.ParticipantSet(k),demographicBreakdown.NMetadata(k), ...
            demographicBreakdown.Male(k),demographicBreakdown.Female(k), ...
            demographicBreakdown.Other(k),demographicBreakdown.MeanAge(k), ...
            demographicBreakdown.SDAge(k));
    end
    save(fullfile(outputDir,"Exp"+experiment+"_stats.mat"),'result');
end
results.supplementaryTable = combined;
results.existingTests = existingCombined;
results.demographics = demographics;
writetable(combined,fullfile(outputDir,'supplementary_statistics.csv'));
save(fullfile(outputDir,'all_stats.mat'),'results');
fprintf('\nSaved supplementary_statistics.csv and MATLAB results to %s\n',outputDir);
fprintf('Check Status before manuscript use; CIs/p-values are unadjusted.\n');
end

function summary = demographic_summary(root,experiment)
participantFolder = fullfile(root,'participant_data');
oldFile = metadata_file(participantFolder,experiment,"metadata_with_headphones");
newFile = fullfile(participantFolder,char("Exp"+experiment+"_metadata_headphone_test.csv"));
allFile = fullfile(participantFolder,char("Exp"+experiment+"_metadata_all.csv"));

oldMetadata = readtable(oldFile);
if isfile(newFile)
    newMetadata = readtable(newFile);
else
    newMetadata = oldMetadata([],:);
end
if isfile(allFile)
    allMetadata = readtable(allFile);
else
    allMetadata = [oldMetadata; newMetadata];
end

oldSummary = summarize_one_set(experiment,"Original",oldMetadata,true);
newSummary = summarize_one_set(experiment,"Headphone test",newMetadata,false);
allSummary = summarize_all_set(experiment,oldSummary,newSummary,allMetadata);
summary = [oldSummary; newSummary; allSummary];
end

function file = metadata_file(participantFolder,experiment,suffix)
file = fullfile(participantFolder,char("Exp"+experiment+"_"+suffix+".csv"));
if ~isfile(file) && suffix=="metadata_with_headphones"
    file = fullfile(participantFolder,char("Exp"+experiment+"_metadata.csv"));
end
assert(isfile(file), 'Missing metadata file for Experiment %s.', experiment);
end

function summary = summarize_all_set(experiment,oldSummary,newSummary,allMetadata)
summary = summarize_one_set(experiment,"All",allMetadata,false);
if oldSummary.NMetadata + newSummary.NMetadata == summary.NMetadata
    summary.Male = oldSummary.Male + newSummary.Male;
    summary.Female = oldSummary.Female + newSummary.Female;
    summary.Other = oldSummary.Other + newSummary.Other;
end
end

function summary = summarize_one_set(experiment,participantSet,metadata,usePaperCounts)
if nargin < 4, usePaperCounts = false; end
if height(metadata)==0
    meanAge = NaN;
    sdAge = NaN;
    male = 0;
    female = 0;
    other = 0;
else
    gender = upper(strtrim(string(metadata.Gender)));
    male = sum(gender=="M" | gender=="MALE");
    female = sum(gender=="F" | gender=="FEMALE");
    other = height(metadata)-male-female;
    meanAge = mean(metadata.Age,'omitnan');
    sdAge = std(metadata.Age,'omitnan');
    if usePaperCounts
        [hasPaperCounts,paperN,paperMale,paperFemale,paperOther] = paper_demographic_counts(experiment);
        if hasPaperCounts && height(metadata)==paperN
            male = paperMale;
            female = paperFemale;
            other = paperOther;
        end
    end
end
Experiment = string(experiment);
ParticipantSet = string(participantSet);
NMetadata = height(metadata);
Male = male;
Female = female;
Other = other;
MeanAge = meanAge;
SDAge = sdAge;
summary = table(Experiment,ParticipantSet,NMetadata,Male,Female,Other,MeanAge,SDAge);
end

function [hasPaperCounts,n,male,female,other] = paper_demographic_counts(experiment)
% "Other" combines
% nonbinary and declined/no-answer responses because the summary table only
% has male/female/other columns.
hasPaperCounts = true;
switch char(experiment)
    case '0'
        n = 51; male = 23; female = 27; other = 1;
    case '1'
        n = 87; male = 40; female = 45; other = 2;
    case '2'
        n = 57; male = 26; female = 30; other = 1;
    case '3a'
        n = 101; male = 53; female = 44; other = 4;
    case '3b'
        n = 118; male = 62; female = 52; other = 4;
    case '3c'
        n = 60; male = 27; female = 30; other = 3;
    case '4'
        n = 55; male = 22; female = 30; other = 3;
    otherwise
        hasPaperCounts = false;
        n = NaN; male = NaN; female = NaN; other = NaN;
end
end

function finalTable = final_comparisons(result)
finalTable = result.table;
finalTable.PValueLowerBound = finalTable.PValue;
finalTable.PValueUpperBound = finalTable.PValue;
if isfield(result,'resamplingDiagnostics')
    diagnostics = result.resamplingDiagnostics;
    for k = 1:height(diagnostics)
        row = finalTable.Condition==diagnostics.Condition(k) & finalTable.Reference=="Recorded";
        assert(nnz(row)==1, 'Expected one Recorded comparison per diagnostic.');
        finalTable.PValueLowerBound(row) = diagnostics.PValueLowerBound(k);
        finalTable.PValueUpperBound(row) = diagnostics.PValueUpperBound(k);
    end
end
if isfield(result,'existingResamplingDiagnostics')
    diagnostics = result.existingResamplingDiagnostics;
    for k = 1:height(diagnostics)
        row = finalTable.Condition==diagnostics.Condition(k) & finalTable.Reference=="Full generative model";
        assert(nnz(row)==1, 'Expected one Full-model comparison per diagnostic.');
        finalTable.PValueLowerBound(row) = diagnostics.PValueLowerBound(k);
        finalTable.PValueUpperBound(row) = diagnostics.PValueUpperBound(k);
    end
end
end
