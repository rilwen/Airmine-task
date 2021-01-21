import numpy as np
import pandas as pd
import unittest

import airmine


class TestAirmine(unittest.TestCase):

    def setUp(self):
        # LHR: 51°28'14"N, 0°27'42"W
        # SYD: 33°56'45"S, 151°10'37"E
        self.places = pd.DataFrame(
            {'Latitude': [51 + (28 / 60) + (14 / 3600), - (33 + 56 / 60 + 45 / 3600)],
             'Longitude': [- (27 / 60 + 42 / 3600), 151 + 10 / 60 + 37 / 3600]},
            index=['LHR', 'SYD'])
        self.distance = 17016

    def test_great_circle_distance(self):
        lhr = self.places.loc['LHR'] * np.pi / 180
        syd = self.places.loc['SYD'] * np.pi / 180
        distance = airmine.great_circle_distance(
            lhr['Latitude'], lhr['Longitude'], syd['Latitude'], syd['Longitude'])
        self.assertAlmostEqual(self.distance, distance,
                               delta=self.distance * 0.001)

    def test_pairs_and_great_circle_distances(self):
        pairs, distances = airmine.pairs_and_great_circle_distances(
            self.places)
        self.assertEqual(1, len(pairs))
        self.assertEqual(1, len(distances))
        self.assertAlmostEqual(self.distance, distances[0],
                               delta=self.distance * 0.001)
        p1, p2 = pairs[0]
        self.assertEqual('LHR', p1.Index)
        self.assertEqual(self.places.loc['LHR', 'Latitude'], p1.Latitude)
        self.assertEqual(self.places.loc['LHR', 'Longitude'], p1.Longitude)
        self.assertEqual('SYD', p2.Index)
        self.assertEqual(self.places.loc['SYD', 'Latitude'], p2.Latitude)
        self.assertEqual(self.places.loc['SYD', 'Longitude'], p2.Longitude)


if __name__ == '__main__':
    unittest.main()
