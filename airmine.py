"""Solution for Airmine code test.

Program calculates great circle distances between all unique pairs of places. Places are either
loaded from the file 'places.csv' or randomly generated. Great circle distance is calculated
using the spherical approximation. The CSV file must be in the same directory as the Python
script file.

Usage: airmine.py [n].
    n: (Optional) number of places to be generated randomly. If provided, n must be integer and >= 2.
"""
import itertools
from typing import List, Optional, Tuple, Union
import sys
import numpy as np
import pandas as pd


EARTH_RADIUS = 6371  # Earth radius using spherical approximation [km].


def generate_places(n: int) -> pd.DataFrame:
    """Generates n randomly located places.

    Args:
        n: Number of places.

    Returns:
        See `get_data`.
    """
    long, lat = np.random.rand(n) * 360 - 180, np.random.rand(n) * 180 - 90
    places = ['Place ' + str(i) for i in range(n)]
    return pd.DataFrame({'Name': places, 'Latitude': long, 'Longitude': lat}).set_index('Name')


def get_data(n: Optional[int] = None) -> pd.DataFrame:
    """Gets data from "places.csv" (if n is None) or randomly generated. 

    Args:
        n: Optional number of places.

    Returns:
        Dataframe of length `n` with columns 'Latitude' and 'Longitude', indexed by place name. Coordinates
        are in degrees. Dataframe index has name 'Name'.
   """
    if n is None:
        df = pd.read_csv("places.csv", index_col=0)
    else:
        df = generate_places(n)
    return df


# Custom type annotation for vectorised functions.
ScalarOrVector = Union[float, np.ndarray]


def great_circle_distance(
        lat1: ScalarOrVector, long1: ScalarOrVector,
        lat2: ScalarOrVector, long2: ScalarOrVector) -> ScalarOrVector:
    """Calculates the great circle distance between two places on Earth, or pairs of such places.

    Vectorised implementation: takes as arguments either scalar coordinates or Numpy arrays of coordinates.
    Uses spherical approximation for Earth geometry. Coordinates should be in radians.

    Args:
        lat1: Latitude(s) of first place(s).
        long1: Longitude(s) of first place(s). 
        lat2: Latitude(s) of second place(s).
        long2: Longitude(s) of second place(s).        

    Returns:
        Great circle distance on Earth or a Numpy array of those.
    """
    return EARTH_RADIUS * (np.arccos(np.sin(lat1) * np.sin(lat2) +
                                     np.cos(lat1) * np.cos(lat2) * np.cos(long1 - long2)))


# Type returned by Pandas itertuples function when iterating over places.
Place = Tuple[str, float, float]


def pairs_and_great_circle_distances(df: pd.DataFrame) -> Tuple[List[Tuple[Place, Place]], np.ndarray]:
    """Calculates the great circle distance between all unique pairs of different places.

    Args:
        df: Dataframe containing place coordinates and names, with columns 'Latitude' and 'Longitude' and indexed by name.
            Coordinates should be in degrees.

    Returns:
        Tuple of:
            1. List of tuples (place 1, place 2), where place 1 != place 2 and each unordered pair appears only once.
            Each place is described by a Pandas named tuple with fields 'Index', 'Latitude' and 'Longitude'.
            2. Numpy array of corresponding great circle distances.
    """
    pairs = list(itertools.combinations(df.itertuples(), 2)
                 )  # find all unique pairs of places
    coordinates = np.array(
        [[p1.Latitude, p1.Longitude, p2.Latitude, p2.Longitude] for p1, p2 in pairs])
    coordinates /= (180 / np.pi)
    return pairs, great_circle_distance(*tuple(coordinates.T))


def main():
    """Main function."""

    # By default, load places from 'places.csv'.
    n = None
    if len(sys.argv) > 1:
        # Handle command line arguments.
        try:
            n = int(sys.argv[1])
        except ValueError:
            sys.exit('Number of places must be an integer, got {}'.format(
                sys.argv[1]))
        if n < 2:
            # Handle inputs for which no place pairs exist.
            sys.exit(
                'Number of places must be at least 2, got {:d}'.format(n))

    data = get_data(n)

    pairs, distances = pairs_and_great_circle_distances(data)

    # How much space to allocate for each name column:
    lnwidth = max(map(len, data.index)) + 2

    average_distance = distances.mean()
    # Find the pair with a distance closest to the average distance.
    closest_pair_index = np.argmin(abs(average_distance - distances))
    closest_pair = pairs[closest_pair_index]

    # Print nicely formatted pairs and great circle distances.
    format_str = '{0:<%d}\t{1:<%d}\t{2:>10.1f} km' % (lnwidth, lnwidth)
    for p, d in zip(pairs, distances):
        print(format_str.format(p[0].Index, p[1].Index, d))
    # Print average distance results.
    print('Average distance: {:.1f} km. Closest pair: {} – {} {:.1f} km.'.format(
        average_distance, closest_pair[0].Index, closest_pair[1].Index, distances[closest_pair_index]))


if __name__ == '__main__':
    main()
