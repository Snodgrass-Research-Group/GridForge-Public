%% Export EVERY variable of EVERY .mat under "KCL-Compliant Grid Generator"
% (all subfolders, recursively) to CSV, so the Python migration never
% depends on MATLAB-only formats. Requires R2019a+ (writematrix/writecell).
%
% Run ONCE in MATLAB from the "KCL-Compliant Grid Generator" folder:
%   >> run('..\gridforge\tools\export_opaque_tables.m')
%
% Output mirrors the folder tree (avoids name collisions between the many
% ConductorData.mat copies):
%   exports\<subfolders>\<matfile>__<variable>.csv
%
% Type handling:
%   table / timetable       -> writetable
%   numeric / logical       -> writematrix (complex: __real.csv + __imag.csv)
%   cell                    -> writecell
%   scalar struct           -> recurse into fields (names joined with __)
%   prob.* fitdist objects  -> 2-row CSV: distribution class + parameters
%   anything else           -> listed at the end as skipped
%
% Then re-run the Python migration:
%   python -m gridforge.io.matlab_import

rootDir = pwd;
outRoot = fullfile(rootDir, 'exports');
files = dir(fullfile(rootDir, '**', '*.mat'));

nExported = 0;
nSkipped = 0;
for f = 1:numel(files)
    inDir = files(f).folder;
    if contains(inDir, [filesep 'exports'])
        continue                                 % never re-export our output
    end
    relDir = erase(inDir, rootDir);
    relDir = regexprep(relDir, ['^\' filesep], '');
    outDir = fullfile(outRoot, relDir);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    [~, base] = fileparts(files(f).name);

    try
        S = load(fullfile(inDir, files(f).name));
    catch err
        fprintf('SKIP (load failed) %s: %s\n', files(f).name, err.message);
        nSkipped = nSkipped + 1;
        continue
    end

    vars = fieldnames(S);
    for v = 1:numel(vars)
        [ok, skip] = export_variable(S.(vars{v}), ...
            fullfile(outDir, [base '__' vars{v}]));
        nExported = nExported + ok;
        nSkipped = nSkipped + skip;
    end
end
fprintf('Done: %d CSVs written to exports\\, %d items skipped.\n', ...
    nExported, nSkipped);
fprintf('Now re-run:  python -m gridforge.io.matlab_import\n');


function [nOk, nSkip] = export_variable(val, stem)
%EXPORT_VARIABLE Write one loaded variable (recursing into scalar structs).
nOk = 0;
nSkip = 0;
try
    if istable(val)
        writetable(val, [stem '.csv']);
        nOk = 1;
    elseif istimetable(val)
        writetable(timetable2table(val), [stem '.csv']);
        nOk = 1;
    elseif isnumeric(val) || islogical(val)
        if isreal(val)
            writematrix(val, [stem '.csv']);
            nOk = 1;
        else                                     % e.g. ComplexLoads
            writematrix(real(val), [stem '__real.csv']);
            writematrix(imag(val), [stem '__imag.csv']);
            nOk = 2;
        end
    elseif iscell(val)
        writecell(val, [stem '.csv']);
        nOk = 1;
    elseif isstruct(val) && isscalar(val)
        flds = fieldnames(val);
        for i = 1:numel(flds)
            [o, s] = export_variable(val.(flds{i}), [stem '__' flds{i}]);
            nOk = nOk + o;
            nSkip = nSkip + s;
        end
    elseif isa(val, 'prob.ProbabilityDistribution')
        pn = val.ParameterNames;                 % e.g. fitdist Normal objects
        pv = num2cell(val.ParameterValues);
        C = [[{'distribution'}, pn(:)']; [{class(val)}, pv(:)']];
        writecell(C, [stem '.csv']);
        nOk = 1;
    elseif ischar(val) || isstring(val)
        writecell(cellstr(val), [stem '.csv']);
        nOk = 1;
    else
        fprintf('SKIP (type %s) %s\n', class(val), stem);
        nSkip = 1;
    end
catch err
    fprintf('SKIP (write failed) %s: %s\n', stem, err.message);
    nSkip = nSkip + 1;
end
end
