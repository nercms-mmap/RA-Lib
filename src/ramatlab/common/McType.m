classdef McType
    % McType: A class representing different types of Markov chains.
    %
    % This class provides constants corresponding to various types of
    % Markov chains used in the model. The types are imported from a
    % Python module.

    properties (Constant)
        % Constants representing different types of Markov chains accessed directly from Python
        MC1 = py.importlib.import_module('src.rapython.unsupervised.markovchain').McType.MC1;
        % Represents the first type of Markov chain.

        MC2 = py.importlib.import_module('src.rapython.unsupervised.markovchain').McType.MC2;
        % Represents the second type of Markov chain.

        MC3 = py.importlib.import_module('src.rapython.unsupervised.markovchain').McType.MC3;
        % Represents the third type of Markov chain.

        MC4 = py.importlib.import_module('src.rapython.unsupervised.markovchain').McType.MC4;
        % Represents the fourth type of Markov chain.
    end
end
