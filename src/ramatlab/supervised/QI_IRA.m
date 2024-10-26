%{
    QI_IRA - Class implementing Quantum-Inspired Interactive Ranking Aggregation (QI-IRA) for ranking tasks.

    This class provides a MATLAB interface to the Python QI-IRA implementation, enabling the execution
    of the QI-IRA method for ranking aggregation based on user-defined parameters.

    Reference:
    ----------
    - Hu, C., Zhang, H., Liang, C., & Huang, H. (2024, March). QI-IRA: Quantum-Inspired Interactive Ranking Aggregation for Person Re-identification. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 3, pp. 2202-2210).

    Author:
    -------
    Qi Deng

    Date:
    -----
    2024-10-18
%}

classdef QI_IRA
    methods(Static)
        function qi_ira(input_file_path, output_file_path, input_rel_path, k_set, iteration, error_rate, input_type)
            % qi_ira - Execute Quantum-Inspired Interactive Ranking Aggregation (QI-IRA) using specified input and output file paths.
            %
            % This function loads data from CSV files, processes the data into a numerical format, and
            % applies the QI-IRA aggregation method to produce the final ranked results.
            %
            % Parameters:
            % -----------
            % input_file_path : str
            %     Path to the input CSV file containing query, voter name, item code, and item score.
            %
            % output_file_path : str
            %     Path to the output CSV file where the ranked results will be saved.
            %
            % input_rel_path : str
            %     Path to the input CSV file containing query, item code, and relevance.
            %
            % input_type : InputType, optional
            %     The type of input data to determine how the item score is interpreted (RANK or SCORE).
            %     Default is InputType.SCORE.
            %
            % Returns:
            % --------
            % None
            %     The function saves the ranked results directly to the specified output file path.

            % Call the Python qi_ira function
            py.importlib.import_module('src.rapython.supervised.qi_ira').qi_ira(input_file_path, output_file_path, input_rel_path, py.int(k_set), py.int(iteration), error_rate, input_type);
        end
    end
end
