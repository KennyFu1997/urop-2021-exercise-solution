from track_generator.command import Command


class ChromosomeElem:

    def __init__(self, command: Command, value: float):
        self.command = command
        self.value = value

    def __str__(self):
        return '({}, {})'.format(self.command.name, self.value)

