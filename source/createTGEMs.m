%% Adjust TGEM to temperature range

baseline_model = "models/TGEM.mat";
% For temp in  10:40
for temp = 10:40
    load(baseline_model);
    kcatAdjModel = adjustTGEM(TGEM,temp,20,true);
    save(sprintf("models/TGEMAdj_%s.mat",int2str(temp)),"kcatAdjModel")
end

function TGEMAdj = adjustTGEM(TGEM, temp, kcat_scaling)
% The function adjusts turnover numbers to a given temperature using the 
% MMRT model described by Hobbs et al. (2013).
% 
% Essentially, this is just a wrapper for Philip's adjustKcatsMMRTGECKO
% function and scales some values. So it relies on the AraTCore package by 
% Philipp.
% 
% Input:
%           struct TGEM:            GECKO-formatted metabolic model with additional fields:
%                                   DH, DS, DCp (obtained by running fitMMRTGECKOModel)
%           double temp:            Temperatur [C] to which the turnover
%                                   rates are adjusted (use 20C as default)
%           double kcat_scaling:    scaling factor for multiplication of 
%                                   adjusted kcat values for enzymes where
%                                   no MMRT fit is available (use 20 as
%                                   default)
%
% Output:   struct TGEMAdj:         Temperature adjusted model


% add sink reaction for water (for modelling transpiration)
TGEM.TRANS_ID = 'Sink_H2O';
if ismember('H2O[e]', TGEM.mets)
    water_id = 'H2O[e]';
elseif ismember('H2O[c]', TGEM.mets)
    water_id = 'H2O[c]';
elseif ismember('H2O_e', TGEM.mets)
    % for GECKO-formatted model
    water_id = 'H2O_e';
elseif ismember('H2O_c', TGEM.mets)
    % for GECKO-formatted model
    water_id = 'H2O_c';
else
    water_idx = find(ismember(TGEM.metFormulas, 'H2O'));
    water_idx_ext = ~cellfun(@isempty, regexp(TGEM.mets(water_idx), '[\[_]e\]?$'));
    water_idx_cyt = ~cellfun(@isempty, regexp(TGEM.mets(water_idx), '[\[_]c\]?$'));
    if any(water_idx_ext)
        water_id = TGEM.mets{water_idx(water_idx_ext)};
    elseif any(water_idx_cyt)
        water_id = TGEM.mets{water_idx(water_idx_cyt)};
    else
        error('Could not find water metabolite ID.')
    end
end

% attempt to find an existing sink reaction for water
water_sink_idx = TGEM.S(findMetIDs(TGEM, water_id),:)<0 & sum(TGEM.S~=0)==1;

if sum(water_sink_idx) == 1
    fprintf('Using the following reaction for transpiration modelling: %s\n',...
        TGEM.rxns{water_sink_idx})
    TGEM.TRANS_ID = TGEM.rxns(water_sink_idx);
elseif sum(water_sink_idx) > 1
    fprintf('Multiple sink reactions found for water:\n')
    disp(TGEM.rxns(water_sink_idx))
    water_sink_idx = find(water_sink_idx);
    fprintf('Choosing reaction the following reaction for transpiration modelling: %s\n',...
        TGEM.rxns{water_sink_idx(1)})
    TGEM.TRANS_ID = TGEM.rxns(water_sink_idx(1));
    fprintf('Blocking remaining water sink reactions\n')
    TGEM.ub(water_sink_idx(2:end)) = 0;
else
    fprintf('No water sink reaction was found, adding sink reaction\n')
    TGEM = addReaction(TGEM, TGEM.TRANS_ID,...
        'reactionFormula', [water_id ' ->']);
    
end

% convert stoichiometric matrix from sparse to full
TGEM.S = full(TGEM.S);

% in case the H2O sink reaction is blocked, change upper bound to 1000
TGEM.ub(findRxnIDs(TGEM, TGEM.TRANS_ID)) = 1000;

% block water sink reaction in another compartments
water_idx = find(ismember(TGEM.metFormulas, 'H2O'));
water_sink_idx = any(TGEM.S(water_idx,:)<0) & sum(TGEM.S~=0)==1 & ~ismember(TGEM.rxns, TGEM.TRANS_ID)';
fprintf('Blocking remaining water sink reactions:\n')
disp(TGEM.rxns(water_sink_idx))
TGEM.ub(water_sink_idx) = 0;



% increase the upper bounds of CO2 and photon import reactions
TGEM.ub(findRxnIDs(TGEM, TGEM.CO2_IMP_ID)) = 1e4;
TGEM.ub(findRxnIDs(TGEM, TGEM.ABS_ID)) = 1e4;




%% Adjust total protein content
if ismember('prot_pool_exchange', TGEM.rxns)
    P = TGEM.proteinModel(temp);
    fprintf('Total protein content: %.2f g/gDW\n', P)
    TGEM.ub(findRxnIDs(TGEM, 'prot_pool_exchange')) = P;
end


%% ADJUST KCATS
% adjust kcats to given temperature
tmpModel = TGEM;
tempKelvin = celsius2kelvin(temp);
TGEMAdj = adjustKcatsMMRTGECKO(tmpModel, tempKelvin, [], kcat_scaling);


end