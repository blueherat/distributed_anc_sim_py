clear; clc;

% Path bootstrap: keep matlab_impl packages first on path.
thisDir = fileparts(mfilename('fullpath'));
if ~strcmpi(pwd, thisDir)
    cd(thisDir);
end
addpath(thisDir, '-begin');

repoRoot = fileparts(thisDir);
datasetPath = fullfile(repoRoot, 'python_impl', 'python_scripts', 'strict_equiv_dataset.mat');
if ~isfile(datasetPath)
    error('Strict dataset not found: %s. Please run python_impl/python_scripts/generate_strict_dataset.py first.', datasetPath);
end

S = load(datasetPath);

time = double(S.time(:));
x = double(S.reference_signal);
d = double(S.desired_signal);

if isvector(x)
    x = x(:);
end
if isvector(d)
    d = d(:);
end

refMicIds = uint32(S.ref_ids(:).');
secSpkIds = uint32(S.sec_ids(:).');
errMicIds = uint32(S.err_ids(:).');

secRirs = double(S.sec_rirs);
secRirLengths = double(S.sec_rir_lengths);

fs = double(S.fs(1));
L = double(S.L(1));
mu = double(S.mu(1));

selected_algorithms = ["CFxLMS", "ADFxLMS", "ADFxLMS-BC", "Diff-FxLMS", "DCFxLMS", "CDFxLMS", "MGDFxLMS"];
curveWindowMs = 50;

fprintf('Strict MATLAB selected algorithms: %s\n', join(selected_algorithms, ', '));

mgr = acoustics.PrecomputedRIRManager(refMicIds, secSpkIds, errMicIds, secRirs, secRirLengths);

summary = struct();
summary.dataset = datasetPath;
summary.algorithms = cellstr(selected_algorithms);
summary.fs = fs;
summary.L = L;
summary.mu = mu;
summary.curve_window_ms = curveWindowMs;
summary.runtimes_s = struct();
summary.nse_db_last_1s = struct();
summary.nr_db_last_1s = struct();

nSamples = length(time);
nAlgs = numel(selected_algorithms);
nseCurves = zeros(nSamples, nAlgs);
nrCurves = zeros(nSamples, nAlgs);

for algIdx = 1:nAlgs
    alg = selected_algorithms(algIdx);
    algName = char(alg);
    safeName = matlab.lang.makeValidName(algName);

    switch alg
        case "CFxLMS"
            params.time            = time;
            params.rirManager      = mgr;
            params.L               = L;
            params.mu              = mu;
            params.referenceSignal = x;
            params.desiredSignal   = d;

            tic;
            results = algorithms.CFxLMS(params);
            t = toc;

        case "ADFxLMS"
            node1 = algorithms.ADFxLMS.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(secSpkIds(1)); node1.addErrMic(errMicIds(1));
            node2 = algorithms.ADFxLMS.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(secSpkIds(2)); node2.addErrMic(errMicIds(2));
            node3 = algorithms.ADFxLMS.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(secSpkIds(3)); node3.addErrMic(errMicIds(3));
            node4 = algorithms.ADFxLMS.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(secSpkIds(4)); node4.addErrMic(errMicIds(4));

            net = topology.Network();
            net.addNode(node1); net.addNode(node2); net.addNode(node3); net.addNode(node4);
            net.connectNodes(1, 2); net.connectNodes(1, 4);
            net.connectNodes(2, 3); net.connectNodes(2, 4);
            net.connectNodes(3, 4);

            params.time            = time;
            params.rirManager      = mgr;
            params.network         = net;
            params.L               = L;
            params.referenceSignal = x;
            params.desiredSignal   = d;

            tic;
            results = algorithms.ADFxLMS(params);
            t = toc;

        case "ADFxLMS-BC"
            node1 = algorithms.ADFxLMS_BC.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(secSpkIds(1)); node1.addErrMic(errMicIds(1));
            node2 = algorithms.ADFxLMS_BC.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(secSpkIds(2)); node2.addErrMic(errMicIds(2));
            node3 = algorithms.ADFxLMS_BC.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(secSpkIds(3)); node3.addErrMic(errMicIds(3));
            node4 = algorithms.ADFxLMS_BC.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(secSpkIds(4)); node4.addErrMic(errMicIds(4));

            net = topology.Network();
            net.addNode(node1); net.addNode(node2); net.addNode(node3); net.addNode(node4);
            net.connectNodes(1, 3); net.connectNodes(1, 4);
            net.connectNodes(2, 3); net.connectNodes(2, 4);

            params.time            = time;
            params.rirManager      = mgr;
            params.network         = net;
            params.L               = L;
            params.referenceSignal = x;
            params.desiredSignal   = d;

            tic;
            results = algorithms.ADFxLMS_BC(params);
            t = toc;

        case "Diff-FxLMS"
            node1 = algorithms.Diff_FxLMS.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(secSpkIds(1)); node1.addErrMic(errMicIds(1));
            node2 = algorithms.Diff_FxLMS.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(secSpkIds(2)); node2.addErrMic(errMicIds(2));
            node3 = algorithms.Diff_FxLMS.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(secSpkIds(3)); node3.addErrMic(errMicIds(3));
            node4 = algorithms.Diff_FxLMS.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(secSpkIds(4)); node4.addErrMic(errMicIds(4));

            net = topology.Network();
            net.addNode(node1); net.addNode(node2); net.addNode(node3); net.addNode(node4);
            net.connectNodes(1, 3); net.connectNodes(1, 4);
            net.connectNodes(2, 3); net.connectNodes(2, 4);

            params.time            = time;
            params.rirManager      = mgr;
            params.network         = net;
            params.L               = L;
            params.referenceSignal = x;
            params.desiredSignal   = d;

            tic;
            results = algorithms.Diff_FxLMS(params);
            t = toc;

        case "DCFxLMS"
            node1 = algorithms.DCFxLMS.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(secSpkIds(1)); node1.addErrMic(errMicIds(1));
            node2 = algorithms.DCFxLMS.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(secSpkIds(2)); node2.addErrMic(errMicIds(2));
            node3 = algorithms.DCFxLMS.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(secSpkIds(3)); node3.addErrMic(errMicIds(3));
            node4 = algorithms.DCFxLMS.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(secSpkIds(4)); node4.addErrMic(errMicIds(4));

            net = topology.Network();
            net.addNode(node1); net.addNode(node2); net.addNode(node3); net.addNode(node4);

            params.time            = time;
            params.rirManager      = mgr;
            params.network         = net;
            params.L               = L;
            params.referenceSignal = x;
            params.desiredSignal   = d;

            tic;
            results = algorithms.DCFxLMS(params);
            t = toc;

        case "CDFxLMS"
            node1 = algorithms.CDFxLMS.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(secSpkIds(1)); node1.addErrMic(errMicIds(1));
            node2 = algorithms.CDFxLMS.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(secSpkIds(2)); node2.addErrMic(errMicIds(2));
            node3 = algorithms.CDFxLMS.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(secSpkIds(3)); node3.addErrMic(errMicIds(3));
            node4 = algorithms.CDFxLMS.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(secSpkIds(4)); node4.addErrMic(errMicIds(4));

            net = topology.Network();
            net.addNode(node1); net.addNode(node2); net.addNode(node3); net.addNode(node4);
            net.connectNodes(1, 3); net.connectNodes(1, 4);
            net.connectNodes(2, 3); net.connectNodes(2, 4);
            net.connectNodes(1, 2); net.connectNodes(3, 4);

            params.time            = time;
            params.rirManager      = mgr;
            params.network         = net;
            params.L               = L;
            params.referenceSignal = x;
            params.desiredSignal   = d;

            tic;
            results = algorithms.CDFxLMS(params);
            t = toc;

        case "MGDFxLMS"
            lc = 16;
            node1 = algorithms.MGDFxLMS.Node(1, mu, lc); node1.addRefMic(refMicIds(1)); node1.addSecSpk(secSpkIds(1)); node1.addErrMic(errMicIds(1));
            node2 = algorithms.MGDFxLMS.Node(2, mu, lc); node2.addRefMic(refMicIds(2)); node2.addSecSpk(secSpkIds(2)); node2.addErrMic(errMicIds(2));
            node3 = algorithms.MGDFxLMS.Node(3, mu, lc); node3.addRefMic(refMicIds(3)); node3.addSecSpk(secSpkIds(3)); node3.addErrMic(errMicIds(3));
            node4 = algorithms.MGDFxLMS.Node(4, mu, lc); node4.addRefMic(refMicIds(4)); node4.addSecSpk(secSpkIds(4)); node4.addErrMic(errMicIds(4));

            net = topology.Network();
            net.addNode(node1); net.addNode(node2); net.addNode(node3); net.addNode(node4);
            net.connectNodes(1, 3); net.connectNodes(1, 4);
            net.connectNodes(2, 3); net.connectNodes(2, 4);
            net.connectNodes(1, 2); net.connectNodes(3, 4);

            params.time            = time;
            params.rirManager      = mgr;
            params.network         = net;
            params.L               = L;
            params.referenceSignal = x;
            params.desiredSignal   = d;

            tic;
            results = algorithms.MGDFxLMS(params);
            t = toc;
    end

    e = results.err_hist;

    fprintf('%s strict-mat runtime: %.6f s.\n', algName, t);
    summary.runtimes_s.(safeName) = t;

    win = min(round(1 * fs), size(e, 1));
    dSeg = d(end-win+1:end, :);
    eSeg = e(end-win+1:end, :);
    dPow = mean(dSeg.^2, 1) + eps;
    ePow = mean(eSeg.^2, 1) + eps;
    nseLast = 10 * log10(ePow ./ dPow);
    nrLast = -nseLast;

    summary.nse_db_last_1s.(safeName) = nseLast;
    summary.nr_db_last_1s.(safeName) = nrLast;

    [nseCurve, nrCurve] = computeConvergenceCurve(d, e, fs, curveWindowMs);
    nseCurves(:, algIdx) = nseCurve;
    nrCurves(:, algIdx) = nrCurve;
end

summaryJson = jsonencode(summary);
summaryPath = fullfile(repoRoot, 'python_impl', 'python_scripts', 'strict_mat_summary.json');
fid = fopen(summaryPath, 'w');
if fid == -1
    error('Cannot write strict summary: %s', summaryPath);
end
fprintf(fid, '%s', summaryJson);
fclose(fid);

curvesPath = fullfile(repoRoot, 'python_impl', 'python_scripts', 'strict_mat_curves.mat');
algorithms = cellstr(selected_algorithms);
save(curvesPath, 'time', 'algorithms', 'nseCurves', 'nrCurves');

fprintf('Strict MATLAB summary saved to: %s\n', summaryPath);
fprintf('Strict MATLAB curves saved to: %s\n', curvesPath);


function [nseCurve, nrCurve] = computeConvergenceCurve(d, e, fs, windowMs)
win = max(1, round((windowMs / 1000) * fs));
kernel = ones(win, 1) / win;

dInst = mean(d.^2, 2);
eInst = mean(e.^2, 2);

dPow = conv(dInst, kernel, 'same') + eps;
ePow = conv(eInst, kernel, 'same') + eps;

nseCurve = 10 * log10(ePow ./ dPow);
nrCurve = -nseCurve;
end
