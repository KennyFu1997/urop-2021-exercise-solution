import math
from typing import List

from euclid import Vector3, Quaternion

from ga.chromosome_elem import ChromosomeElem
from track_generator.state import State
from track_generator.track_point import TrackPoint
from track_generator.track_script import TrackScript


def generate_track(chromosome_elements: List[ChromosomeElem]) -> List[TrackPoint]:
    track_points: List[TrackPoint] = []

    start_position = Vector3(x=49.7, y=0.5, z=50.0)
    span = Vector3(x=0.0, y=0.0, z=0.0)
    span_dist = 2

    track_script = TrackScript(chromosome_elements=chromosome_elements)
    if track_script.parse_chromosome():

        dy = 0.0

        s = start_position
        s.y = 0.5
        span.x = 0.0
        span.y = 0.0
        span.z = span_dist
        turn_val = 10.0

        for track_script_element in track_script.track:

            if track_script_element.state == State.AngleDY:
                turn_val = track_script_element.value
            elif track_script_element.state == State.CurveY:
                dy = track_script_element.value * turn_val
            else:
                dy = 0.0

            for i in range(track_script_element.num_to_set):
                turn = dy
                rot = Quaternion.new_rotate_euler(math.radians(turn), 0, 0)
                span = rot * span.normalized()
                span *= span_dist
                s = s + span

                track_points.append(TrackPoint(x=s.x, y=s.z))

    return track_points
