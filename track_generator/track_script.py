from typing import List

from ga.chromosome_elem import ChromosomeElem
from track_generator.command import Command
from track_generator.state import State
from track_generator.track_script_elem import TrackScriptElem


class TrackScript:

    def __init__(self, chromosome_elements: List[ChromosomeElem]):
        self.track: List[TrackScriptElem] = []
        self.chromosome_elements = chromosome_elements

    def parse_chromosome(self) -> bool:

        for ce in self.chromosome_elements:

            command = ce.command
            args = ce.value

            tse = TrackScriptElem()

            if command == Command.S:
                tse.state = State.Straight
                tse.value = 1.0
                tse.num_to_set = int(args)
            elif command == Command.L:
                tse.state = State.CurveY
                tse.value = -1.0
                tse.num_to_set = int(args)
            elif command == Command.R:
                tse.state = State.CurveY
                tse.value = 1.0
                tse.num_to_set = int(args)
            elif command == Command.DY:
                tse.state = State.AngleDY
                tse.value = float(args)
                tse.num_to_set = 0
            else:
                raise NotImplementedError('Command {} not found'.format(command.name))

            self.track.append(tse)

        return len(self.track) > 0
