classdef PrecomputedRIRManager < handle
    %PRECOMPUTEDRIRMANAGER Minimal manager for strict-equivalence runs.
    %
    % Provides the subset of RIRManager API required by ANC algorithms:
    % - ReferenceMicrophones / SecondarySpeakers / ErrorMicrophones dictionaries
    % - getSecondaryRIR(spkId, micId)

    properties
        ReferenceMicrophones dictionary
        SecondarySpeakers dictionary
        ErrorMicrophones dictionary
    end

    properties (Access = private)
        SecondaryRIRs dictionary
    end

    methods
        function obj = PrecomputedRIRManager(refIds, secIds, errIds, secRirs, secRirLengths)
            refIds = uint32(refIds(:).');
            secIds = uint32(secIds(:).');
            errIds = uint32(errIds(:).');

            if ndims(secRirs) ~= 3
                error('secRirs must be a 3D array [numSec, numErr, maxLen].');
            end

            if ~isequal(size(secRirLengths), [numel(secIds), numel(errIds)])
                error('secRirLengths must have size [numel(secIds), numel(errIds)].');
            end

            if size(secRirs, 1) ~= numel(secIds) || size(secRirs, 2) ~= numel(errIds)
                error('secRirs first two dimensions must match secIds and errIds lengths.');
            end

            obj.ReferenceMicrophones = configureDictionary('uint32', 'cell');
            obj.SecondarySpeakers = configureDictionary('uint32', 'cell');
            obj.ErrorMicrophones = configureDictionary('uint32', 'cell');
            obj.SecondaryRIRs = configureDictionary('string', 'cell');

            % Placeholder coordinates keep compatibility with existing algorithm code.
            for id = refIds
                obj.ReferenceMicrophones(id) = {[0 0 0]};
            end
            for id = secIds
                obj.SecondarySpeakers(id) = {[0 0 0]};
            end
            for id = errIds
                obj.ErrorMicrophones(id) = {[0 0 0]};
            end

            for i = 1:numel(secIds)
                for j = 1:numel(errIds)
                    nTaps = max(0, double(secRirLengths(i, j)));
                    if nTaps == 0
                        rir = 0;
                    else
                        rir = reshape(secRirs(i, j, 1:nTaps), 1, []);
                    end
                    key = "S" + string(secIds(i)) + "->M" + string(errIds(j));
                    obj.SecondaryRIRs(key) = {rir};
                end
            end
        end

        function h = getSecondaryRIR(obj, spkId, micId)
            key = "S" + string(spkId) + "->M" + string(micId);
            if ~isKey(obj.SecondaryRIRs, key)
                error('Secondary RIR for path (%s) does not exist.', key);
            end
            h = cell2mat(obj.SecondaryRIRs(key));
        end
    end
end
