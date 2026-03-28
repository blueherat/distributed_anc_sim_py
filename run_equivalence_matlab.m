clear; clc;

selected_algorithms = ["CFxLMS", "ADFxLMS", "ADFxLMS-BC", "Diff-FxLMS", "DCFxLMS", "CDFxLMS", "MGDFxLMS"];
fprintf('当前选择运行的算法: %s\n', join(selected_algorithms, ', '));

mgr = acoustics.RIRManager();
mgr.Fs = 4000;
mgr.Room = [5 5 5];
mgr.Algorithm = "image-source";
mgr.ImageSourceOrder = 2;
mgr.MaterialAbsorption = .5;
mgr.MaterialScattering = 0.07;

center = mgr.Room / 2;
mgr.addPrimarySpeaker(101, center + [1 0 0]);

rRef = 0.9;
refMicIds = uint32([401 402 403 404]);
mgr.addReferenceMicrophone(refMicIds(1), center + [rRef 0 0]);
mgr.addReferenceMicrophone(refMicIds(2), center - [rRef 0 0]);
mgr.addReferenceMicrophone(refMicIds(3), center + [0 rRef 0]);
mgr.addReferenceMicrophone(refMicIds(4), center - [0 rRef 0]);

r1 = 0.6;
mgr.addSecondarySpeaker(201, center + [r1 0 0]);
mgr.addSecondarySpeaker(202, center - [r1 0 0]);
mgr.addSecondarySpeaker(203, center + [0 r1 0]);
mgr.addSecondarySpeaker(204, center - [0 r1 0]);

r2 = 0.3;
mgr.addErrorMicrophone(301, center + [r2 0 0]);
mgr.addErrorMicrophone(302, center - [r2 0 0]);
mgr.addErrorMicrophone(303, center + [0 r2 0]);
mgr.addErrorMicrophone(304, center - [0 r2 0]);

fprintf('正在构建声学环境...\n');
mgr.build(false);

rng(42);
duration = 2;
f_low = 100;
f_high = 1500;
[noise, time] = utils.wn_gen(mgr.Fs, duration, f_low, f_high);

sourceSignal = noise ./ max(abs(noise), [], 1);
d = mgr.calculateDesiredSignal(sourceSignal, length(time));
x = mgr.calculateReferenceSignal(sourceSignal, length(time));
refScale = max(abs(x), [], 1);
refScale(refScale < eps) = 1;
x = x ./ refScale;

L = 1024;
mu = 1e-4;

summary = struct();
summary.algorithms = cellstr(selected_algorithms);
summary.duration_s = duration;
summary.fs = mgr.Fs;
summary.f_low = f_low;
summary.f_high = f_high;
summary.L = L;
summary.mu = mu;
summary.runtimes_s = struct();
summary.nse_db_last_1s = struct();

for alg = selected_algorithms
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
            node1 = algorithms.ADFxLMS.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(201); node1.addErrMic(301);
            node2 = algorithms.ADFxLMS.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(202); node2.addErrMic(302);
            node3 = algorithms.ADFxLMS.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(203); node3.addErrMic(303);
            node4 = algorithms.ADFxLMS.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(204); node4.addErrMic(304);

            net = topology.Network();
            net.addNode(node1); net.addNode(node2); net.addNode(node3); net.addNode(node4);
            net.connectNodes(1, 2); net.connectNodes(1, 4);
            net.connectNodes(2, 3); net.connectNodes(2, 4);
            net.connectNodes(1, 2); net.connectNodes(3, 4);

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
            node1 = algorithms.ADFxLMS_BC.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(201); node1.addErrMic(301);
            node2 = algorithms.ADFxLMS_BC.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(202); node2.addErrMic(302);
            node3 = algorithms.ADFxLMS_BC.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(203); node3.addErrMic(303);
            node4 = algorithms.ADFxLMS_BC.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(204); node4.addErrMic(304);

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
            node1 = algorithms.Diff_FxLMS.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(201); node1.addErrMic(301);
            node2 = algorithms.Diff_FxLMS.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(202); node2.addErrMic(302);
            node3 = algorithms.Diff_FxLMS.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(203); node3.addErrMic(303);
            node4 = algorithms.Diff_FxLMS.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(204); node4.addErrMic(304);

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
            node1 = algorithms.DCFxLMS.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(201); node1.addErrMic(301);
            node2 = algorithms.DCFxLMS.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(202); node2.addErrMic(302);
            node3 = algorithms.DCFxLMS.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(203); node3.addErrMic(303);
            node4 = algorithms.DCFxLMS.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(204); node4.addErrMic(304);

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
            node1 = algorithms.CDFxLMS.Node(1, mu); node1.addRefMic(refMicIds(1)); node1.addSecSpk(201); node1.addErrMic(301);
            node2 = algorithms.CDFxLMS.Node(2, mu); node2.addRefMic(refMicIds(2)); node2.addSecSpk(202); node2.addErrMic(302);
            node3 = algorithms.CDFxLMS.Node(3, mu); node3.addRefMic(refMicIds(3)); node3.addSecSpk(203); node3.addErrMic(303);
            node4 = algorithms.CDFxLMS.Node(4, mu); node4.addRefMic(refMicIds(4)); node4.addSecSpk(204); node4.addErrMic(304);

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
            node1 = algorithms.MGDFxLMS.Node(1, mu, lc); node1.addRefMic(refMicIds(1)); node1.addSecSpk(201); node1.addErrMic(301);
            node2 = algorithms.MGDFxLMS.Node(2, mu, lc); node2.addRefMic(refMicIds(2)); node2.addSecSpk(202); node2.addErrMic(302);
            node3 = algorithms.MGDFxLMS.Node(3, mu, lc); node3.addRefMic(refMicIds(3)); node3.addSecSpk(203); node3.addErrMic(303);
            node4 = algorithms.MGDFxLMS.Node(4, mu, lc); node4.addRefMic(refMicIds(4)); node4.addSecSpk(204); node4.addErrMic(304);

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

    fprintf('%s 仿真耗时 %f 秒。\n', algName, t);
    summary.runtimes_s.(safeName) = t;

    win = min(round(1 * mgr.Fs), size(results.err_hist, 1));
    dSeg = d(end-win+1:end, :);
    eSeg = results.err_hist(end-win+1:end, :);
    dPow = mean(dSeg.^2, 1) + eps;
    ePow = mean(eSeg.^2, 1) + eps;
    nse = 10 * log10(ePow ./ dPow);
    summary.nse_db_last_1s.(safeName) = nse;
end

jsonText = jsonencode(summary);
outPath = fullfile('python_scripts', 'equivalence_matlab_summary.json');
fid = fopen(outPath, 'w');
if fid == -1
    error('无法写入输出文件: %s', outPath);
end
fprintf(fid, '%s', jsonText);
fclose(fid);

fprintf('结果已写入: %s\n', outPath);
fprintf('全部完成。\n');
