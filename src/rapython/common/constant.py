from enum import Enum, auto

__all__ = ['InputType']


class InputType(Enum):
    """
    InputType enum class representing different input types for the model.

    Attributes
    ----------
    RANK : auto()
        Represents the rank-based input type.
    SCORE : auto()
        Represents the score-based input type.
    """
    RANK = auto()
    SCORE = auto()

    @staticmethod
    def check_input_type(input_type):
        """
        Validate and convert the input parameter to the InputType enum.

        Parameters
        ----------
        input_type : InputType or other
            An input parameter that should be of type InputType. If it is not,
            the function will attempt to convert it based on the name attribute.

        Returns
        -------
        InputType
            A valid InputType enum member.

        Raises
        ------
        ValueError
            If input_type cannot be converted to a valid InputType enum member.
        """

        # Check if input_type is already an InputType instance
        if isinstance(input_type, InputType):
            return input_type  # Return it directly if it is a valid InputType enum

        # Try to convert input_type to InputType if it has a 'name' attribute
        try:
            param = InputType[input_type.name]
        except (KeyError, AttributeError):
            # Raise a ValueError with a descriptive message if conversion fails
            raise ValueError(
                f"Invalid input_type: {input_type}. Must be one of {[m.name for m in InputType]}"
            )

        return param
