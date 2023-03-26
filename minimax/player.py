class Player:
    """A class that represents a player in a game"""
    def __init__(self, name: int) -> None:
        """
        Initializes a player.
        Parameters:
            name: a single-character string to represent the player in textual representations of game state
        """
        self.name = name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Player):
            return False
        return self.name == other.name
