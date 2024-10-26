classdef IRAType
    % IRAType: A class representing different IRA methods.
    %
    % This class provides constants corresponding to IRA methods that can be
    % used in the model. The IRA methods are imported from a Python module.

    properties (Constant)
        % Constants representing different IRA methods accessed directly from Python
        IRA_RANK = py.importlib.import_module('src.rapython.supervised.ira').MethodType.IRA_RANK;
        % Uses the rank-based IRA method.

        IRA_SCORE = py.importlib.import_module('src.rapython.supervised.ira').MethodType.IRA_SCORE;
        % Uses the score-based IRA method.
    end
end
