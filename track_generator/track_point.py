
class TrackPoint:

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y)
