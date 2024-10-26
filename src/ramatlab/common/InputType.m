classdef InputType
    % InputType: A class representing different input types for the model.
    %
    % This class provides constants corresponding to input types that can be
    % used in the model. The input types are imported from a Python module.

    properties (Constant)
        % Constants representing different input types accessed directly from Python
        RANK = py.importlib.import_module('src.rapython.common.constant').InputType.RANK;
        % Represents the rank-based input type.

        SCORE = py.importlib.import_module('src.rapython.common.constant').InputType.SCORE;
        % Represents the score-based input type.
    end
end
