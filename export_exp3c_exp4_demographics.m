function demographics = export_exp3c_exp4_demographics(outputDir)
% EXPORT_EXP3C_EXP4_DEMOGRAPHICS Export participant counts and age summaries.
%   demographics = export_exp3c_exp4_demographics
%   demographics = export_exp3c_exp4_demographics(outputDir)
%
% The CSV reports original, headphone-test, and combined participant counts
% plus age summaries for Experiments 3c and 4, with did-not-answer separated
% from other gender responses.
root = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outputDir) || strlength(string(outputDir))==0
    outputDir = fullfile(root,'stats_results');
end
if ~isfolder(outputDir), mkdir(outputDir); end
participantFolder = fullfile(root,'participant_data');

demographics = [ ...
    summarize_participants("3c","Original",readtable(fullfile(participantFolder,'Exp3c_metadata_with_headphones.csv'))); ...
    summarize_participants("3c","Headphone test",readtable(fullfile(participantFolder,'Exp3c_metadata_headphone_test.csv'))); ...
    summarize_participants("3c","All",readtable(fullfile(participantFolder,'Exp3c_metadata_all.csv'))); ...
    summarize_participants("4","Original",readtable(fullfile(participantFolder,'Exp4_metadata_with_headphones.csv'))); ...
    summarize_participants("4","Headphone test",readtable(fullfile(participantFolder,'Exp4_metadata_headphone_test.csv'))); ...
    summarize_participants("4","All",readtable(fullfile(participantFolder,'Exp4_metadata_all.csv'))) ...
    ];

writetable(demographics,fullfile(outputDir,'Exp3c_Exp4_demographics_for_Josh.csv'));
end

function summary = summarize_participants(experiment,participantSet,metadata)
gender = upper(regexprep(strtrim(string(metadata.Gender)),'[\s_]',''));
male = sum(gender=="M" | gender=="MALE");
female = sum(gender=="F" | gender=="FEMALE");
didNotAnswer = sum(gender=="DIDNOTANSWER" | gender=="DECLINED" | ...
    gender=="DECLINEDTOANSWER" | gender=="NOANSWER" | ...
    gender=="NOTANSWERED" | gender=="NA" | gender=="N/A" | ...
    gender=="PREFERNOTTOSAY");
other = height(metadata)-male-female-didNotAnswer;

Experiment = string(experiment);
ParticipantSet = string(participantSet);
N = height(metadata);
M = male;
F = female;
Other = other;
DidNotAnswer = didNotAnswer;
MeanAge = mean(metadata.Age,'omitnan');
SDAge = std(metadata.Age,'omitnan');
summary = table(Experiment,ParticipantSet,N,M,F,Other,DidNotAnswer,MeanAge,SDAge);
end
