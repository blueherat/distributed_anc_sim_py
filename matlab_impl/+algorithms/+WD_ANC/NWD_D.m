function results = NWD_D(params)
%% 1. 解包参数
time            = params.time;
rirManager      = params.rirManager;
x               = params.referenceSignal; % 参考信号
d               = params.desiredSignal;   % 期望信号
block_size      = params.block_size;
overlap         = params.overlap;
hop             = block_size - overlap;

% 从rirManager获取参数
keySecSpks = keys(rirManager.SecondarySpeakers);
keyErrMics = keys(rirManager.ErrorMicrophones);

numRefMics      = numEntries(rirManager.ReferenceMicrophones);
numErrMics      = numEntries(rirManager.ErrorMicrophones);

if size(x, 2) ~= numRefMics
    error('referenceSignal columns (%d) must match number of reference microphones (%d).', size(x, 2), numRefMics);
end

nSamples = length(time);

%% 2. 初始化
max_Ls_hat = 0;
for i = keySecSpks'
    Ls_hat = length(rirManager.getSecondaryRIR(i, keyErrMics(1)));
    if Ls_hat > max_Ls_hat
        max_Ls_hat = Ls_hat;
    end
end

e = d; % 误差信号初始化

x_buffer = zeros(block_size, numRefMics);

e_buffer = zeros(block_size, numErrMics);
%% 3. 主循环
disp('开始NWD_D仿真...');
for n = 1:hop:nSamples - block_size
    x_buffer = [x_buffer(hop+1:end, :); x(n:n+hop-1, :)];
    e_buffer = [e_buffer(hop+1:end, :); e(n:n+hop-1, :)];
end

results.err_hist = e;
results.filter_coeffs = dictionary;

end